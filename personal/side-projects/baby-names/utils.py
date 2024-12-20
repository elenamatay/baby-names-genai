import vertexai
import json
import asyncio # Not needed in a notebook, only used when runnning in a standalone Python program, as there  you need to start the asyncio event loop yourself 
import os
from vertexai.generative_models import GenerativeModel, SafetySetting
from google.cloud import storage
from collections import defaultdict

PROJECT_ID="your_project_id" # @param - GCP Project ID
LOCATION="your_location" # @param - Region where resources will run and be deployed
MODEL="your_selected_model" # @param - Selected LLM - Notebook runs successfully with "gemini-1.5-flash-002"
BUCKET_ID="your_bucket_id" # @param - GCS Bucket ID to store master JSON file in

SAFETY_SETTINGS = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

# PART 1: Helper functions for DB Generation

prompt = """You are a helpful assistant specializing in baby name analysis. You provide detailed information about names, including their meaning, origin, sound details, and other relevant facts.  You also suggest similar names that people might find appealing.

Instructions:

1. Analyze the list of baby names provided in the input.
2. For each name, research and extract the following information. Add all the info you can find in your knowledge base for each name:
    * Meaning: Provide a concise explanation of the name's meaning.
    * Origin: Identify the name's linguistic and geographical origin. Should be an array containing one or more of this list: African, American, Arabic, Aramaic, Armenian, Australian, Babylonian, Basque, Brazilian, British, Bulgarian, Cambodian, Celtic, Chinese, Czech, Danish, Dutch, Egyptian, Filipino, Finnish, French, Gaelic, German, Ghanaian, Greek, Haitian, Hawaiian, Hebrew, Hungarian, Icelandic, Indian, Irish, Italian, Jamaican, Japanese, Korean, Latin, Mexican, Native American, Nigerian, Norse, Norwegian, Pacific Islander, Persian, Polish, Portuguese, Romanian, Russian, Sanskrit, Scandinavian, Scottish, Slavic, Spanish, Swahili, Swedish, Swiss, Thai, Turkish, Ukranian, Uncertain, Vietnamese, Welsh, Yid.
    * Sound Details: Describe the name's phonetic structure, including phonemes and syllables.
    * Variants: If any available, suggest 1-8 variants of the name. Otherwise return null.
    * Famous: Provide a list of up to 6 diverse famous or inspiring people with the same name (if any, otherwise don't provide this list)  
    * Other Info: Include any other relevant information or details, such as popularity trends, historical figures associated with the name, or cultural significance.
    * Family meaning: Highlight how this name's meaning could relate to a this baby's family members personality, and to the ambition on how they envision the child's personality and values.
    * Likely Liked: If you know it, include 3-10 other distinct names that people who like this name also like. They may or not share similar characteristics or appeal. Never suggest more than 10 names.
3. If the input list is empty, return the following message: "Please provide a list of baby names."
4. Only if you encounter a name for which you cannot find information, return null as the JSON content for that specific name. Same if you don't have any information on a specific field, e.g. famous people, for that name.
5. Format your output as a JSON object, where each name is a key, and its corresponding information is the value. Each value should be a JSON object containing the fields "meaning," "origin," "sound_details," "variants," "famous", and "other_info". NEVER repeat information or suggested names inside one same name entry.

IMPORTANT: NEVER add " characters inside any value of any JSON, when you want to reflect a meaning or anything else. Alwats use ` instead of " to avoid conflicts.

Example:
Input: ["Alice", "Ajani", "Bob", "Elena", "Bass", "Bernardine"]
Output:
{
  "Alice": {
    "meaning": "noble, and exalted or light",
    "origin": ["German", "French"],
    "original_form": ["Adelheidis (Germanic)",  "Adelais (Old French)"],
    "sound_details": {
      "phonemes": ["æ", "l", "ɪ", "s"],
      "syllables": 2
    },
    "variants": ["Alicia", "Alison", "Adelaide"],
    "famous": ["Alice - Princess Alice of the UK, daughter of Queen Victoria and Prince Albert", "Alice Eve - British Actress", "Alice Krige - Sudafrican Actress", "Alice Marble - American Tennis Player", "Alice Walker - American Novelist, Poet"],
    "other_info": "Alice is a classic feminine name of German origin. It acts as a short version of the Germanic name Adalheidis, meaning `noble` and `exalted`. Popularized by Lewis Carroll's Alice in Wonderland, Alice hit the naming charts at the turn of the 20th century and has remained a beloved contender ever since. An everlasting darling heroine, Alice was also picked as a baby name by Tina Fey in 2005 and Theodore Roosevelt in the 19th century. The name Alice has touched hearts throughout history but now it’s loved enough that parents are straining against societal norms!",
    "family_meaning": "Likely to be creative, artistic, and perhaps a bit unconventional family. They might be drawn to literature, history, or the arts. They value intelligence, imagination, and a strong sense of individuality. A child with this name is envisioned as a thoughtful, curious, and independent thinker, possibly pursuing a career in writing, music, or the arts.",
    "likely_liked": ["Violet", "Lucy", "Audrey", "Amelia", "Scarlett", "Alexander", "Liam", "Jasper", "Henry", "Benjamin"]
  },
  "Ajani": {
    "meaning": "He who wins the struggle",
    "origin": ["African"],
    "original_form": ["Ajani (African)"]
    "sound_details": {
        "phonemes": ["æ", "dʒ", "ɑː", "n", "i"],
        "syllables": 3
    },
    "variants": ["Adjani"],
    "famous": null,
    "other_info": "Ajani (pronounced ah-JAH-nee) derives from African origins and holds significant meaning. It is derived from the Yoruba language, a major ethnic group in Nigeria, West Africa. In Yoruba culture, names carry deep meaning, often reflecting the aspirations, beliefs, or experiences of individuals. Ajani specifically means `He Who Wins the Struggle` and embodies the spirit of triumph and resilience.Throughout history, the name Ajani has been associated with individuals who have endured and conquered various challenges. In ancient Africa, the name may have been bestowed upon warriors, leaders, or individuals who exhibited remarkable determination in overcoming adversity. Today, the name Ajani continues to carry this powerful symbolism and is cherished among the African diaspora globally. In modern times, Ajani can be found as a given name among individuals of African heritage. It has gained popularity as parents embrace traditional African names and seek to instill a sense of heritage and strength in their children. The name's significance resonates with many, as it represents the struggles faced throughout history and the ability to triumph over them. As a result, Ajani has made its mark not only in African communities but also in multicultural societies where diverse names are celebrated.",
    "family_meaning": "A family with this baby name is likely to be strong and resilient, ready to overcome any challenge that life brings. Parents might envision their child as a determined individual who will strive for their goals. The name carries the message that even in the face of difficulties, the child will emerge victorious. Ultimately, the career path that Ajani chooses will depend on their personal interests, skills, and passions. However, his name can serve as a source of inspiration and motivation, encouraging him to embrace challenges, strive for excellence, leadership roles, and make a positive impact on the world.",
    "likely_liked": ["Kimoni", "Asante", "Jeremiah", "Ajamu", "Keoni", "Amir"]
  },
  "Bob": {
    "meaning": "bright fame",
    "origin": ["British", "German"],
    "original_form": "Robert (Germanic)",
    "sound_details": {
      "phonemes": ["b", "ɑː", "b"],
      "syllables": 1
    },
    "variants": ["Robert", "Bobby", "Bert"],
    "famous":["Bob Ross - Painter", "Bob Marley - Jamaican Reggae singer", "Bob Saget - Actor", "Bob Reese - Freerunner", "Bob Dylan - Folk Singer"],
      "other_info": "Bob was originated from the hypocorism Rob, short for Robert. Rhyming names were popular in the Middle Ages, so Richard became Rick, Hick, or Dick, William became Will, Gill, or Bill, and Robert became Rob, Hob, Dob, Nob, or Bob. From Old High German Hrodebert, a compound of Hruod (Old Norse: Hróðr) meaning `fame`, `glory`, `honour`, `praise`, `renown`, `godlike` and Berht  meaning `bright`, `light`, `shining`. After becoming widely used in Continental Europe, the name entered England in its Old French form Robert, where an Old English cognate form (Hrēodbēorht, Hrodberht, Hrēodbēorð, Hrœdbœrð, Hrœdberð, Hrōðberχtŕ) had existed before the Norman Conquest.",
    "family_meaning": "A family with this baby name is likely to be down-to-earth, practical, and have a good sense of humor. This family might be drawn to outdoor activities, sports, or manual labor. They value honesty, loyalty, and a strong work ethic. A baby with this name is envisioned as reliable, hardworking, and a good friend, possibly pursuing a career in construction, engineering, or a trade.",
    "likely_liked": ["Adam", "Ben", "Jack", "Aaron", "Phillip", "Mae", "Leila", "June", "Lauren", "Jasmine"]
  },
  "Elena": {
    "meaning": "torch, bright, shining light, or resplendent", 
    "origin": ["Greek"],
    "original_form": "Helena (Greek)",
    "sound_details": {
      "phonemes": ["ɛ", "l", "ɛ", "n", "ə"],
      "syllables": 3
    },
    "variants": ["Helena", "Alena", "Alenka"; "Alyona", "Elene", "Helen", "Hélène", "Eliana"],
    "famous": ["Helena of Troy: A legendary figure from Greek mythology whose beauty sparked the Trojan War. Origin of this name.","Elena Ferrante - Pseudonymous Italian novelist known for her Neapolitan Novels.","Elena Kagan - American jurist who serves as an Associate Justice of the Supreme Court of the United States.", "Elena Poniatowska - Mexican journalist and writer who has won numerous awards for her work."],
    "other_info": "Elena is the latin variant of the name `Helena` (in Greek: Ἑλένη) is a feminine given name of Greek origin. In Greek ἑλένη -heléne- means `torch`, so it is also commonly translated as bright, dazzling -shining light- or resplendent. It is also associated with the concept of the `most resplendent woman in the world` derived from the Trojan myth of Helena, a woman of incomparable beauty who was persuaded by Paris to elope with her. From this same story it also acquires the meaning of `beloved woman`. Elena is the latin version of Helena (original greek version). Elena is a classic name with a rich history. It's associated with elegance, grace, and beauty. It's been a popular choice for centuries and continues to be a beloved name.",
    "family_meaning": "Often drawn to classic elegance and timeless beauty. They might be sophisticated, cultured, and have a love for history or mythology. This baby's family values grace, intelligence, and a strong sense of self. A baby with this name is envisioned  as poised, confident, and successful, perhaps pursuing a career in law, diplomacy, or business.",
    "likely_liked": ["Amelia", "Ella", "Aria", "Charlotte", "Sophia", "William", "Oliver", "Lucas", "Liam", "Ethan"]
  },
  "Bass": null,
  "Bernadine": {
    "meaning": "brave, strong, bold, and courageous",
    "origin": ["German", "French"],
    "original_form": "Bernard (German)",
    "sound_details": {
        "phonemes": ["bɜː", "n", "ə", "d", "iː", "n"],
        "syllables": 3
    },
    "variants": ["Bernadette", "Bernarda", "Bernardina", "Berny"],
    "famous": [ "Bernadine Healy - American physician and former director of the National Institutes of Health", "Bernadine Michelle Bezuidenhout - South African cricketer", "Bernadine Williams - Main character of the drama film `Clemency`, played by the American actress Alfre Woodard"],
    "other_info": "Bernadine is the french feminine form of Bernard, which is a German name meaning `brave`, `strong`, `bold`, and `courageous`. It is a popular name in many countries, including the United States, Canada, and the United Kingdom. It is a classic name with a strong and elegant sound. It is a good choice for parents who are looking for a name that is both traditional and sophisticated.",
    "family_meaning": "For a family choosing the name "Bernadine" for their baby girl, they likely envision a sophisticated and strong-willed individual. The name's French origin lends it an air of elegance and refinement, while its meaning, `strong as a bear`, suggests a resilient and courageous spirit. They might hope their daughter will grow up to be a confident and independent woman, capable of achieving great things. Given the name's connotations of strength, elegance, and a touch of old-world charm, potential career paths for a Bernadine might include creative fields (fashion, interior design, or art, where her refined taste and artistic flair can shine), law, diplomacy, or politics, business and finance. Ultimately, the specific career path will depend on Bernadine's individual passions and talents, but the name itself can inspire a sense of confidence, grace, and a determination to succeed.",
    "likely_liked": ["Lottee", "Bera", "Eleanor", "Beatrice", "Josephine", "Claire", "Alexander", "Benjamin", "William"]
  },
  "Jerald": {
        "meaning": "ruler of the spear",
        "origin": ["German"],
        "sound_details": {
            "phonemes": [
                "dʒ",
                "ɛ",
                "r",
                "ə",
                "l",
                "d"
            ],
            "syllables": 2
        },
        "variants": ["Gerald", "Gerold", "Jarald", "Jarold"],
        "famous": [
            "Jerald Brown - Canadian football player",
            "Jerald Daemyon - American jazz musician",
            "Jerald `Jerry` Sloan - American basketball coach"
        ],
        "other_info": "Jerald is a masculine given name of German origin. It is a variant of the name `Gerald`, which is derived from the Germanic elements `ger` meaning `spear` and `wald` meaning `ruler`. The name Jerald has been in use for centuries and has a strong and classic sound. It is a good choice for parents who are looking for a name that is both traditional and sophisticated.",
        "family_meaning": "A family choosing the name Jerald for their baby boy likely envisions a strong, determined, and perhaps even a bit adventurous individual. The name's meaning, `ruler of the spear`, suggests a leader, a protector, and someone who is not afraid to take charge. They might hope their son will grow up to be a confident and successful man, perhaps pursuing a career in law, politics, or the military. The name Jerald can inspire a sense of ambition, courage, and a desire to make a difference in the world.",
        "likely_liked": ["Jared", "Gerald", "Ethan", "Owen", "Liam", "Noah", "Caleb", "Jackson", "William", "Benjamin"]
    }
}
"""

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
    "response_mime_type": "application/json",
    "response_schema": {"type": "OBJECT", "properties": {"response": {"type": "STRING"}}},
}

async def generate_names_info_async(baby_names, index):
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL)
    
    response = await model.generate_content_async(
        [prompt, 
         f"""
          Your turn:
          Input: {baby_names}
          Output:
          """],
        generation_config=generation_config,
        safety_settings=SAFETY_SETTINGS,
        stream=False,
    )

    # Ensure the processed directory exists, save files
    processed_dir = os.path.join('../data/generated')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Determine the letter from the first name in the list
    if baby_names:
        first_letter = baby_names[0][0].upper()
        file_path = f"../data/generated/G_{first_letter}{index + 1}.json"
    else:
        file_path = f"../data/generated/G_{index + 1}.json"
    
    with open(file_path, 'w') as json_file:
        json.dump(response.text, json_file, indent=4, ensure_ascii=False)

    print(f"JSON data for list {index + 1} has been written to {file_path}")
    return response.text


async def process_all_lists(lists_of_names):
    tasks = [generate_names_info_async(names, i) for i, names in enumerate(lists_of_names)]
    results = await asyncio.gather(*tasks)
    return results


def clean_json_files(directory):
    # Ensure the processed directory exists
    processed_dir = os.path.join(directory, 'cleaned')
    print(f"Creating processed directory at: {processed_dir}")
    os.makedirs(processed_dir, exist_ok=True)
    
    # List all files in the directory
    all_files = os.listdir(directory)
    print(f"All files in directory: {all_files}")
    
    # Iterate over each file in the directory
    for filename in all_files:
        print(f"Checking file: {filename}")
        if filename.startswith('G_') and filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")
            
            # Read the content of the file
            with open(file_path, 'r') as file:
                content = file.read()
                print(f"Read content from {file_path}: {content}")
            
            # Parse the JSON content
            try:
                # Clean the response content
                response_content_fixed = content.replace('\\"', '"').replace('\\n', '').replace('\\', '').replace('"{', '{').replace('}"', '}')
                print(f"Cleaned response content: {response_content_fixed}")

                outer_json = json.loads(response_content_fixed)
                print(f"Parsed outer JSON with type {type(outer_json)}: {outer_json}")

                # Ensure outer_json is a dictionary
                if isinstance(outer_json, str):
                    print("Outer JSON is a string, parsing again.")
                    outer_json = json.loads(outer_json)
                if not isinstance(outer_json, dict):
                    raise ValueError("Parsed outer JSON is not a dictionary")

                # Check if "response" key exists
                if "response" not in outer_json:
                    print(f"Key 'response' not found in {filename}")
                    continue

                # Extract the response content
                response_content = outer_json["response"]
                print(f"Extracted response content: {response_content}")

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"Error processing file {filename}: {e}")
                continue
            
            # Convert the extracted response content into a pretty format
            pretty_json = json.dumps(response_content, indent=4, ensure_ascii=False)
            print(f"Pretty JSON content: {pretty_json}")

            # Save the pretty JSON to a new file with `_clean` appended to the original file name
            new_filename = f"{os.path.splitext(filename)[0]}_clean.json"
            new_file_path = os.path.join(processed_dir, new_filename)
            print(f"Saving pretty JSON to {new_file_path}")
            with open(new_file_path, 'w') as new_file:
                new_file.write(pretty_json)
                print(f"Saved pretty JSON to {new_file_path}")
            
            print(f"Processed and saved: {new_file_path}")

def get_created_names(directory):
    created_names = set()
    repeated_names = defaultdict(list)
    
    # Iterate through all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                # Extract the first name from the JSON file
                first_name = list(data.keys())[0]
                if first_name in created_names:
                    repeated_names[first_name].append(filename)
                else:
                    created_names.add(first_name)
                    repeated_names[first_name].append(filename)
    
    return created_names, repeated_names

def filter_remaining_girl_names(all_girl_names_list, created_names):
    remaining_girl_names = []
    
    # Iterate through all sublists in all_girl_names_list
    for sublist in all_girl_names_list:
        if sublist and sublist[0] not in created_names:
            remaining_girl_names.append(sublist)
    
    return remaining_girl_names

def process_remaining_girl_names(directory, all_girl_names_list):
    # Get the list of created names and repeated names
    created_names, repeated_names = get_created_names(directory)

    # Get the remaining girl names
    remaining_girl_names = filter_remaining_girl_names(all_girl_names_list, created_names)

    # Print the results
    print("Created lists start with names:", created_names)
    print("Remaining lists start with names:", remaining_girl_names)
    print("Total girl names yet to generate data for:", len(remaining_girl_names))
    
    # Print repeated names and their corresponding files
    print("Repeated names and their corresponding files:")
    for name in sorted(repeated_names.keys()):
        if len(repeated_names[name]) > 1:
            print(f"{name}: {repeated_names[name]}")

    return remaining_girl_names

def merge_json_files_by_letter(directory_path):
    # Dictionary to hold merged data for each starting letter
    merged_data = defaultdict(dict)
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            # Get the starting letter of the file
            starting_letter = filename[0].upper()
            
            # Read the JSON file
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Merge the data into the corresponding letter's dictionary
                if isinstance(data, dict):
                    for key, value in data.items():
                        merged_data[starting_letter][key] = value
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                merged_data[starting_letter][key] = value
    
    # Write the merged data to new JSON files
    for letter, data_dict in merged_data.items():
        output_file = os.path.join(directory_path, f'{letter}_master_initial.json')
        with open(output_file, 'w') as file:
            json.dump(data_dict, file, indent=4, ensure_ascii=False)


def add_attributes_field(master_json_path, output_json_name):
    # Load the master JSON
    with open(master_json_path, 'r') as file:
        data = json.load(file)
    
    # Define the words to search for in the meaning, family_meaning, and other_info fields
    attribute_words = {
        "Religious": ["god", "faith", "religion", "angel"],
        "Classic": ["tradition", "classic", "traditional"],
        "Modern": ["modern"],
        "Popular": ["popular", "extended", " common"],
        "Edgy": ["uncommon", "edgy"],
        "Earthy": ["nature", "earth", "living beings", "earthy"],
        "Historical": ["historical", "historic", "epic"]
    }
    
    # Iterate through each name in the JSON
    for name, details in data.items():
        # Skip the $schema and type keys
        if name in ["$schema", "type"]:
            continue
        
        # Skip entries with null values
        if details is None or not isinstance(details, dict):
            continue
        
        meaning = details.get("meaning", "")
        family_meaning = details.get("family_meaning", "")
        other_info = details.get("other_info", "")
        
        # Initialize the attributes list
        attributes = []
        
        # Check if any of the attribute words are in the meaning, family_meaning, or other_info fields
        for attribute, words in attribute_words.items():
            if (meaning and any(word in meaning.lower() for word in words)) or \
               (family_meaning and any(word in family_meaning.lower() for word in words)) or \
               (other_info and any(word in other_info.lower() for word in words)):
                attributes.append(attribute)
        
        # Add the attributes field to the details
        details["attributes"] = attributes
    
    # Determine the output path
    output_json_path = os.path.join(os.path.dirname(master_json_path), output_json_name)
    
    # Save the modified JSON to the output path
    with open(output_json_path, 'w') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def remove_duplicates(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = remove_duplicates(value)
    elif isinstance(data, list):
        # Remove duplicates while preserving order
        seen = set()
        new_list = []
        for item in data:
            item_tuple = tuple(item.items()) if isinstance(item, dict) else item
            if item_tuple not in seen:
                seen.add(item_tuple)
                new_list.append(remove_duplicates(item))
        data = new_list
    return data

def remove_duplicates_from_json(json_path, output_path):
    # Load the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Remove duplicates
    data = remove_duplicates(data)
    
    # Save the modified JSON to the output path
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def contains_keywords(data, keywords):
    if isinstance(data, dict):
        for key, value in data.items():
            if contains_keywords(value, keywords):
                return True
    elif isinstance(data, list):
        for item in data:
            if contains_keywords(item, keywords):
                return True
    elif isinstance(data, str):
        for keyword in keywords:
            if keyword in data.lower():
                return True
    return False

def get_names_with_keywords(data, keywords):
    matching_names = []

    for name, details in data.items():
        if not isinstance(details, dict):
            continue
        
        if contains_keywords(details, keywords):
            matching_names.append(name)
    
    return matching_names

def load_json_and_find_names(json_path, keywords):
    # Load the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Get names with specified keywords
    matching_names = get_names_with_keywords(data, keywords)
    
    return matching_names


# Function to remove duplicates
def remove_duplicate_names(json_path):
    
    with open(json_path, 'r') as file:
        data = json.load(file) # Load the JSON data

    unique_names = {}
    deleted_names = {}

    for name, details in data.items():
        if name not in unique_names:
            unique_names[name] = details
        else:
            if name not in deleted_names:
                deleted_names[name] = 0
            deleted_names[name] += 1

    return unique_names, deleted_names


def keep_names_in_json(json_path, names_to_keep, output_path):
    # Load the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Keep only specified names
    filtered_data = {name: details for name, details in data.items() if name in names_to_keep}
    
    # Save the modified JSON to the output path
    with open(output_path, 'w') as file:
        json.dump(filtered_data, file, indent=2, ensure_ascii=False)

def upload_to_gcs(source_file_name, destination_blob_name):
    """Uploads a file to the bucket and renames it."""
    # Initialize a client
    storage_client = storage.Client(PROJECT_ID)

    # Get the bucket
    bucket = storage_client.bucket(BUCKET_ID)

    # Create a blob object from the new file name
    blob = bucket.blob(destination_blob_name)

    # Upload the file to GCS
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")



# PART 2: Helper functions for Names Generation / Finding