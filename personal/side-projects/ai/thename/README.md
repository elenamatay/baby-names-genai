Baby Names AI Project
Welcome to the Baby Names AI Project! This repository showcases a side project that leverages AI to find the ideal name for an upcoming baby based on their family's preferences. This project is designed to demonstrate the use of AI in a practical application, making it a great addition to your portfolio for AI PM or DevRel roles.

Overview
This project generates synthetic data about baby names, starting with a Kaggle dataset and using Gemini AI to enrich each name with various attributes such as meaning, origin, sound details, famous people, and other useful information. The data is then cleansed and prepared for further use. Based on user input from a questionnaire/form, the project filters the data and makes another call to Gemini to find the most suitable names for a family.

Key Files
girl-names-db.ipynb
This Jupyter notebook outlines the process of creating a dataset for girl names. It includes:

Setup: Imports necessary libraries and sets up the environment.
Helper Functions: Defines functions to generate, process, and clean baby names data.
Data Generation: Uses Gemini AI to generate detailed information about girl names.
Data Cleansing: Cleans the generated data to ensure accuracy and consistency.
Master JSON Creation: Merges individual JSON files into a master JSON file for further use.
generate_names_new.py
This script is responsible for finding the ideal baby names based on user input using Gemini AI. It includes the following key components:

Vertex AI Initialization: Sets up the Vertex AI environment with project details.
Safety Settings: Configures safety settings to filter out harmful content.
User Input Processing: Collects and processes user input from a questionnaire/form.
Data Filtering: Filters the baby names data based on user preferences (e.g., origins, sound details, attributes).
Name Recommendation: Uses Gemini AI to find the most suitable names for a family based on the filtered data.
Output Generation: Creates a JSON output with the top 10 recommended names and detailed explanations for each.
Project Workflow
Data Generation:

Start with a Kaggle dataset of baby names.
Use Gemini AI to enrich each name with detailed information.
Save the generated data to Google Cloud Storage.
Data Cleansing:

Clean the generated data to remove any inconsistencies.
Merge individual JSON files into a master JSON file.
User Input Processing:

Collect user input from a questionnaire/form.
Filter the data based on user preferences (e.g., origins, sound details, attributes).
Name Recommendation:

Use Gemini AI to find the most suitable names for a family based on the filtered data.
Return the top 10 names to the user with detailed explanations.
How to Use
Clone the repository:

Set up the environment:

Install the required dependencies:
Run the Jupyter notebook:

Open girl-names-db.ipynb in Jupyter Notebook or JupyterLab.
Follow the instructions in the notebook to generate and process baby names data.
Deploy the Cloud Function:

Deploy the generate_names_new.py script as a Google Cloud Function.
Configure the function to handle user input and return name recommendations.
Conclusion
This project demonstrates the use of AI to find the ideal baby names based on family preferences. It showcases practical applications of AI and data processing, making it a valuable addition to your portfolio. Feel free to explore the code and adapt it to your needs.