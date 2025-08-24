import os
import json
import requests

def get_github_token_from_env():
    token = os.environ.get('GITHUB_API_TOKEN')
    if not token:
        raise ValueError("FATAL: GITHUB_API_TOKEN environment variable not found!")
    return token

def fetch_repositories(token):
    search_query = "microservice language:python stars:>100"
    headers = {'Authorization': f'token {token}'}
    api_url = f"https://api.github.com/search/repositories?q={search_query}&sort=stars&order=desc&per_page=100"
    
    print(f"Fetching data from GitHub API with query: '{search_query}'...")
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        print("Successfully fetched data from GitHub.")
        return response.json().get('items', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from GitHub: {e}")
        raise

def generate_evaluation_set(repos_data):
    dataset = []
    if not repos_data:
        return dataset
        
    print(f"Generating evaluation set from {len(repos_data)} repositories...")
    for repo in repos_data:
        description = repo.get('description', '') or ''
        if description and len(description.split()) > 5:
            service_details = {
                "func_name": repo.get('name'),
                "docstring": description,
                "url": repo.get('html_url'),
            }
            dataset.append({"query": description, "expected_service": service_details})
    return dataset

if __name__ == "__main__":
    try:
        github_token = get_github_token_from_env()
        repositories = fetch_repositories(github_token)
        
        if repositories:
            evaluation_dataset = generate_evaluation_set(repositories)
            output_filename = 'evaluation_dataset.json'
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(evaluation_dataset, f, indent=4)
                
            print(f"\nSuccess! Created '{output_filename}' with {len(evaluation_dataset)} entries.")
        else:
            print("\nWarning: No repositories were found.")

    except (ValueError, requests.exceptions.RequestException) as e:
        print(e)
        exit(1)