import os
import json
from openai import OpenAI 
import pywikibot
import tiktoken
import networkx as nx
import matplotlib.pyplot as plt
import re
from relations import our_relations
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from queue import Queue
import torch
from memit import MEMITHyperParams, apply_memit_to_model
from edit_request import request
import sys
sys.path.append('/data/maochuanlu/Knowledge_graph/memit')

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pywikibot.config.socket_timeout = 30
SITE = pywikibot.Site("wikidata", "wikidata")
REPO = SITE.data_repository()

def sanitize_input(text):
    """Remove unwanted prefixes and trim text."""
    return re.sub(r'^[\d\.\-]+\s*', '', text).strip()

def robust_request(item_id):
    """Fetch a single item from Wikidata by item ID."""
    try:
        item = pywikibot.ItemPage(REPO, item_id)
        item.get()
        print(f"Successfully fetched Wikidata item: {item_id}")
        return item if item.exists() else None
    except Exception as e:
        print(f"Failed to fetch item '{item_id}' due to error: {e}")
        return None
    
def load_hparams_from_json(json_path):
    with open(json_path, 'r') as file:
        hparams_dict = json.load(file)
    return MEMITHyperParams(**hparams_dict)

def fetch_label_by_id(entity_id):
    try:
        page = pywikibot.PropertyPage(REPO, entity_id) if entity_id.startswith('P') else pywikibot.ItemPage(REPO, entity_id)
        page.get(force=True)
        label = page.labels.get('en', 'No label found')
        print(f"Label for {entity_id}: {label}")
        return label
    except Exception as e:
        print(f"Error fetching label for ID {entity_id}: {e}")
        return "Invalid ID"

def paraphrase_subject(subject_label):
    prompt = (
        "Bill Clinton is also known as:\n"
        "- William Clinton\n"
        "- William Jefferson Clinton\n"
        "- The 42nd president of the United States\n"
        "\n"
        "United Kingdom is also known as:\n"
        "- UK\n"
        "- Britain\n"
        "- England\n"
        "\n"

        f"{subject_label} is also known as:"
    )
    enc = tiktoken.encoding_for_model("gpt-4o")
    prompt_tokens = enc.encode(prompt)
    prompt_token_count = len(prompt_tokens)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate paraphrases for the subject in specific form."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    paraphrases_text = response.choices[0].message.content.strip()
    response_tokens = enc.encode(paraphrases_text)
    response_token_count = len(response_tokens)
    total_tokens_used = prompt_token_count + response_token_count
    paraphrases = re.split(r'\s*\n+', paraphrases_text)
    sanitized_paraphrases = []
    for p in paraphrases:
        if ":" in p:
            p = p.split(":")[1].strip()
        sanitized_paraphrase = sanitize_input(p)
        if is_valid_paraphrase_subject(sanitized_paraphrase):
            sanitized_paraphrases.append(sanitized_paraphrase)
    print(f"Subject paraphrases for '{subject_label}': {sanitized_paraphrases}")
    print(f"Total tokens used: {total_tokens_used}")
    return sanitized_paraphrases, total_tokens_used

def paraphrase_relation(relation_label):
    instructions = [
        f"'{relation_label}' may be described as:",
        f"'{relation_label}' refers to:",
        "please describe '{}' in a few words:".format(relation_label)
    ]

    all_paraphrases = set()  # Use a set to avoid duplicates
    total_tokens = 0
    for instruction in instructions:
        prompt = (
            f"'notable work' may be described as:\n"
            "- A work of great value\n"
            "- A work of importance\n"
            "'notable work' refers to:\n"
            "- Significant achievements\n"
            "- Important contributions\n"
            "please describe 'notable work' in a few words:\n"
            "- Key accomplishments\n"
            "- Major works\n"
            "\n"
            f"{instruction}"
        )
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_token_count = len(prompt_tokens)
        response = client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate paraphrases for the relation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        
        paraphrases_text = response.choices[0].message.content.strip()
        paraphrases = re.split(r'\s*\n+', paraphrases_text)
        response_tokens = enc.encode(paraphrases_text)
        response_token_count = len(response_tokens)
        total_tokens_used = prompt_token_count + response_token_count
        total_tokens += total_tokens_used
        valid_paraphrases = [sanitize_input(p) for p in paraphrases if is_valid_paraphrase_relation(p, instructions)]
        all_paraphrases.update(valid_paraphrases)  # Add to set to avoid duplicates
    print(f"Total tokens used: {total_tokens}")
    print(f"Relation paraphrases for '{relation_label}': {list(all_paraphrases)}")
    return list(all_paraphrases), total_tokens

def is_valid_paraphrase_relation(paraphrase, instructions):
    """Check if the generated paraphrase is valid based on some criteria."""
    paraphrase_lower = paraphrase.lower()
    invalid_phrases = ["can also be defined as", "is also known as", "is referred to as", "also known as", "please paraphrase"]
    for instr in instructions:
        instr_lower = instr.lower()
        if instr_lower in paraphrase_lower or paraphrase_lower in instr_lower or any(phrase in paraphrase_lower for phrase in invalid_phrases):
            return False
    return len(paraphrase) > 0 and not paraphrase_lower.startswith("error")


def is_valid_paraphrase_subject(paraphrase):
    valid = len(paraphrase.split()) > 1 or (len(paraphrase) > 1 and paraphrase.isalpha())
    if not valid:
        print(f"Invalid paraphrase discarded: {paraphrase}")
    return valid


def resolve_wikidata_id(paraphrases):
    wikipedia_site = pywikibot.Site('en', 'wikipedia')
    for paraphrase in paraphrases:
        print(f"Resolving paraphrase: {paraphrase}")
        search_page = wikipedia_site.search(paraphrase, total=1)
        for page in search_page:
            if page.exists():
                if page.isRedirectPage():
                    page = page.getRedirectTarget()
                if page.data_item():
                    wikidata_id = page.data_item().title()
                    print(f"Resolved to Wikidata ID: {wikidata_id} for paraphrase: {paraphrase}")
                    return wikidata_id
        print(f"No Wikidata ID found for paraphrase: {paraphrase}")
    return None

def setup_model(model_name, model_dir, apply_memit=False, memit_hparams=None, requests=None):
    """
    Load the tokenizer and model based on the model name.
    """
    
    if model_name == 'gpt2_xl':
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token 
        model = AutoModelForCausalLM.from_pretrained(model_dir, pad_token_id=tokenizer.eos_token_id).to('cuda')
    elif model_name == 'gpt_j':
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token 
        model = AutoModelForCausalLM.from_pretrained(model_dir, pad_token_id=tokenizer.eos_token_id).to('cuda')
    elif model_name == 'llama2':
        tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token 
        model = LlamaForCausalLM.from_pretrained(model_dir).to('cuda')
    if apply_memit and memit_hparams and requests:
        model, _ = apply_memit_to_model(model, tokenizer, requests, memit_hparams, copy=False)
    return tokenizer, model

def generate_object(entity_label, relation_label, tokenizer, model):
    # Define the prompt
    prompt = f"""Based on the information provided, please answer the following question in strict format:
    Q: Monte Cremasco # country
    A: Italy
    Q: Johnny Depp # children
    A: Jack Depp, Lily-Rose Depp
    Q: Wolfgang Sauseng # employer
    A: University of Music and Performing Arts Vienna
    Q: {entity_label} # {relation_label}
    A:"""

    # Use the pre-edited model passed from the main function
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to('cuda')
    max_length = input_ids.shape[1] + 50
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length).to('cuda')
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("full response: ", full_response)

    # Extract the relevant answer
    query_string = f"Q: {entity_label} # {relation_label}\n    A:"
    start_index = full_response.find(query_string)
    if start_index != -1:
        start_index += len(query_string)
        end_index = full_response.find("\n    Q:", start_index)
        if end_index == -1:  # No more questions, take till end
            end_index = len(full_response)
        answer = full_response[start_index:end_index].strip()
    else:
        answer = "No answer found."

    answer_list = [item.strip() for item in answer.split(',')]
    unique_answers = list(set(answer_list))
    unique_answer_str = ', '.join(unique_answers)

    # Print the answer for debugging
    print("Generated response:", unique_answer_str)
    return unique_answer_str


    
def visualize_graph(graph):
    plt.figure(figsize=(15, 15))  # Reduced figure size

    pos = nx.spring_layout(graph, k=3, iterations=50) 

    labels = {node: node for node in graph.nodes()}
    sizes = [len(str(label)) * 2000 for label in labels.values()]

    for node, size in zip(labels, sizes):
        print(f"Node: {node}, Size: {size}, Label: {labels[node]}")

    nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color='skyblue', edgecolors='black', alpha=0.6)
    nx.draw_networkx_edges(graph, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray')
    nx.draw_networkx_labels(graph, pos, labels, font_size=12)

    edge_labels = nx.get_edge_attributes(graph, 'label')
    for edge, label in edge_labels.items():
        print(f"Edge: {edge}, Label: {label}, Type: {type(label)}")

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', label_pos=0.5)

    plt.title('Knowledge Graph Visualization')
    plt.axis('off')
    
    output_path = '/data/maochuanlu/Knowledge_graph/LBJ_kg.pdf'  # Save as PDF
    try:
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Graph saved to {output_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")

    plt.clf()  # Clear the figure to free up memory
    plt.show()


def fetch_initial_relations(wikidata_item):
    relations = []
    if not wikidata_item:
        return relations
    for claim in wikidata_item.claims:
        target_items = wikidata_item.claims[claim]
        for target_item in target_items:
            target = target_item.getTarget()
            if isinstance(target, pywikibot.ItemPage):
                relations.append((claim, target.title()))
    return relations

def generate_relations(entity_label):
    prompt = (
        "Q: Javier Culson\n"
        "A: participant of # place of birth # sex or gender # country of citizenship # occupation # family name # given name # educated at # sport # sports discipline competed in\n"
        "Q: René Magritte\n"
        "A: ethnic group # place of birth # place of death # sex or gender # spouse # country of citizenship # member of political party # native language # place of burial # cause of death # residence # family name # given name # manner of death # educated at # field of work # work location # represented by\n"
        "Q: Nadym\n"
        "A: country # capital of # coordinate location # population # area # elevation above sea level\n"
        "Q: Stryn\n"
        "A: significant event # head of government # country # capital # separated from\n"
        "Q: 1585\n"
        "A: said to be the same as # follows\n"
        "Q: Bornheim\n"
        "A: head of government # country # member of # coordinate location # population # area # elevation above sea level\n"
        "Q: Aló Presidente\n"
        "A: genre # country of origin # cast member # original network\n"
        f"Q: {entity_label}\n"
        "A:"
    )
    enc = tiktoken.encoding_for_model("gpt-4o")
    prompt_tokens = enc.encode(prompt)
    prompt_token_count = len(prompt_tokens)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the query exactly in the format of the provided examples, listing attributes separated by #."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    relations_text = response.choices[0].message.content.strip()
    relations_text = relations_text.replace('\n', ' ')
    # Encode the response to count the tokens
    response_tokens = enc.encode(relations_text)
    response_token_count = len(response_tokens)

    # Calculate the total number of tokens used
    total_tokens_used = prompt_token_count + response_token_count
    relations = [sanitize_input(r) for r in re.split(r'#\s*', relations_text)]
    print(f"Generated relations for '{entity_label}': {relations}")
    print(f"Total tokens used: {total_tokens_used}")
    return relations, total_tokens_used

def save_graph_to_text_file(graph, output_path):
    """
    Save the relationships in the graph to a text file.
    
    Args:
        graph (networkx.DiGraph): The knowledge graph.
        output_path (str): Path to the output text file.
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        for subject, object_, data in graph.edges(data=True):
            relation = data.get('label', 'has relation')
            line = f"{subject} {relation} {object_}\n"
            file.write(line)
            print(f"Saved relation: {line.strip()}")
    print(f"Graph relationships saved to {output_path}")


def construct_knowledge_graph(entity_label, max_depth=2, branch_limit=2, tokenizer=None, model=None):

    graph = nx.DiGraph()
    queue = Queue()
    queue.put((entity_label, 0))  # Enqueue the initial node and its depth
    total_tokens = 0
    print(f"Starting graph construction with root entity ID: {entity_label}")

    while not queue.empty():
        current_label, current_depth = queue.get()
        print(f"Processing entity: {current_label} at depth: {current_depth}")

        if current_depth > max_depth:
            print("Current depth exceeds max depth, skipping...")
            continue  # Skip processing if the current depth exceeds the maximum depth

        #1. get the subject
        if not graph.has_node(current_label):
            graph.add_node(current_label)
            print(f"Added node: {current_label} at depth {current_depth}")

        if current_depth == max_depth:
            print(f"Reached maximum depth at node: {current_label}, not expanding further.")
            continue  # Do not expand nodes at the maximum depth

        #2. paraphrase subject
        paraphrases, tokens_of_subject = paraphrase_subject(current_label)
        total_tokens += tokens_of_subject
        print(f"Paraphrases found for '{current_label}': {paraphrases}")

        # Initialize dictionary to store relations for each paraphrase
        paraphrase_relations = {paraphrase: set() for paraphrase in paraphrases}
        paraphrase_relations[current_label] = set()

        # Collect all possible relations for each paraphrase
        for paraphrase in paraphrases:
            #3. get the corresponding paraphrased subjects' ids
            paraphrase_id = resolve_wikidata_id([paraphrase])
            print(f"Resolved Wikidata ID for paraphrase '{paraphrase}': {paraphrase_id}")

            if paraphrase_id:
                item = robust_request(paraphrase_id)
                #4. use wikidata to find relation for each paraphrased subject
                relations = fetch_initial_relations(item)
                print(f"Initial relations fetched for paraphrase '{paraphrase}': {relations}")

                #5. collect all relations for each paraphrased subject
                for rel_id, _ in relations:
                    paraphrase_relations[paraphrase].add(rel_id)

        #6. calculate intersection of all relation sets and filter with our_relations
        valid_our_relations = {v for v in our_relations.values() if v}
        print(f"valid relations: {valid_our_relations}")
        val_paraphrased_relations = paraphrase_relations.values()
        print(f"paraphrase_relations.values(): {val_paraphrased_relations}")

        common_relation_ids = set()
        for paraphrase_relation_sets in paraphrase_relations.values():
            common_relations = paraphrase_relation_sets.intersection(valid_our_relations)
            common_relation_ids = common_relation_ids.union(common_relations)

        #common_relation_ids = set.intersection(*paraphrase_relations.values(), valid_our_relations)
        print(f"Common relation IDs across all paraphrases and our_relations: {common_relation_ids}")


        #7. if we cannot find any intersections in these relations then we call generate_relations which call openai to generate relation
        #otherwise, just use fetch_label_by_id to transfer each relation_ids to relation_labels
        if not common_relation_ids:
            print("No common relations found, generating new relations...")
            common_relation_labels, tokens_of_relations = generate_relations(current_label)  
            total_tokens += tokens_of_relations
            print(f"Generated relations: {common_relation_ids}")
        else: 
            common_relation_labels = {fetch_label_by_id(rel_id) for rel_id in common_relation_ids}
        

        #8.for all common_relations, we paraphrase them
        branches_created = 0
        for relation_label in common_relation_labels:
            if branches_created >= branch_limit:
                print("Branch limit reached, not creating more branches.")
                break
            # = paraphrase_relation(relation_label)
            #print(f"Paraphrases for relation '{relation_label}': {relation_paraphrases}")

            #9. for one paraphrased relation and original subject (not paraphrased one), we generate object 
            #NOTE: there are two versions of generating objects, one is only accept objects that were generated by at least two realizations of the relation to improve precision.
            #But if we do this, by openai sometimes will not generate any valid object
            #The other is to pick the first paraphrased_relation to generate object. But that precision will be lower. Here is the second version.
            #First version is commented
            # object_counter = {}
            #for relation_paraphrase in relation_paraphrases:
            generated_object = generate_object(current_label, relation_label, tokenizer,model)
            print(f"generated_object: {generated_object}")
            if not graph.has_edge(current_label, generated_object):
                graph.add_edge(current_label, generated_object, label=relation_label)
                print(f"Added edge from '{current_label}' to '{generated_object}' with relation '{relation_label}' at depth {current_depth}")
                queue.put((generated_object, current_depth + 1))
                branches_created += 1
            #     if generated_object in object_counter:
            #         object_counter[generated_object] += 1
            #     else:
            #         object_counter[generated_object] = 1
            # for generated_object, count in object_counter.items():
            #     if count >= 2 and not graph.has_edge(current_label, generated_object):
            #         graph.add_edge(current_label, generated_object, label=relation_paraphrase)
            #         print(f"Added edge from '{current_label}' to '{generated_object}' with relation '{relation_paraphrase}' at depth {current_depth}")
            #         if current_depth + 1 <= max_depth:
            #             queue.put((generated_object, current_depth + 1))

            #         branches_created += 1
            #         if branches_created >= branch_limit:
            #             print("Branch limit reached, not creating more branches.")
            #             break

    print("Graph construction completed.")
    print("TOTAL TOKEN USED:", total_tokens)
    return graph


def main():
    root_entity_label = "Lebron James"
    max_depth = 2
    model_name = 'gpt2_xl'

    # Step 1: Set up and apply MEMIT to the model
    model_dirs = {
        'gpt2_xl': '/data/akshat/models/gpt2-xl',
        'gpt_j': '/data/akshat/models/gpt-j-6b',
        'llama2': '/data/akshat/models/Llama-2-7b-hf'
    }

    hparams_path = '/data/maochuanlu/Knowledge_graph/memit/hparams/MEMIT/gpt2-xl.json'
    memit_hparams = load_hparams_from_json(hparams_path)

    if model_name in model_dirs:
        tokenizer, model = setup_model(
            model_name, 
            model_dirs[model_name], 
            apply_memit=True, 
            memit_hparams=memit_hparams, 
            requests=request
        )
    else:
        raise ValueError(f"Model {model_name} not found in model_dirs.")

    # Step 2: Pass the edited model to the construct_knowledge_graph function
    graph = construct_knowledge_graph(root_entity_label, max_depth, branch_limit=3, tokenizer=tokenizer, model=model)

    visualize_graph(graph)
    text_output_path = '/data/maochuanlu/Knowledge_graph/relation_text/gpt2_xl_kg.txt'
    save_graph_to_text_file(graph, text_output_path)

if __name__ == "__main__":
    main()
