from google.colab import drive
drive.mount('/content/drive', force_remount = True)

import json
import random

input_file_name = 'data.json'
input_file_path = f"/content/drive/MyDrive/{input_file_name}"
output_file_name = 'preprocessed_data.json'
output_file_path = f"/content/drive/MyDrive/{output_file_name}"

# open JSON file
with open(input_file_path, 'r') as file:
    data = json.load(file)

print(f"Number of objects in the {input_file_name}: {len(data)}\n")

# print the attributes from the first object
if data:
    first_object = data[0]

    print(f"Attributes in the JSON file:")
    # for key, value in first_object.items():
    #     print(f"{key}: {value}")
    for key, value in first_object.items():
        print(key)
else:
    print("No objects found in the JSON file.")

print('')

# list for storing new data
new_dataset = []

length = len(data)

for i in range(length):
    obj = data[i]
    dialogs = obj.get('dialog', [])
    chosen_topic = obj.get('chosen_topic')
    # print(f"Topic: {chosen_topic} / Dialog Length: {len(dialogs)}")

    # extract passages from dialog
    passage_extract = ""
    for dialog_data in dialogs:
        retrieved_passages = dialog_data.get('retrieved_passages', '')
        for passage in retrieved_passages:
            extract = ""
            for passage_topic, passage_sentences in passage.items():
                for sentence in passage_sentences:
                    extract += f"{sentence} "
                extract = extract[:-1] # removing blank at the last
                extract = f"{{{passage_topic}: {extract}}} "
            passage_extract += extract
    passage_extract = passage_extract[:-1] # removing blank at the last

    # make a single string to make a pre-processed data
    result_string = ""
    odd = True
    for dialog_data in dialogs:
        speaker = dialog_data.get('speaker', '')
        speaker = speaker.split('_')[1]
        text = dialog_data.get('text', '')

        if speaker and text:
            if result_string == "":
                result_string += f"<s>[INST] <<SYS>> \
You have just met the other person, who seems quite curious, and you are eager to discuss a topic with them! \
As a wizard, your goal is to inform your conversation partner about a topic that either you or your partner will choose. \
You have an access to paragraphs from Wikipedia possibly relevant to the conversation. \
Before each conversation turn you can read these paragraphs and then potentially base your next reply on that observed knowledge. \
Do not simply parrot this knowledge, but use it to craft a relevant reply, and present any relevant knowledge in a fun and engaging way, if possible. \
Following are paragraphs retrieved from the Wikipedia pages in the form of {{topic: paragraph}}. {passage_extract} \
<</SYS>> {speaker}: {text} "
                odd = False
            else:
                if odd:
                    result_string += f"<s>[INST] {speaker}: {text} "
                    odd = False
                else:
                    result_string += f"[/INST] {speaker}: {text} </s>"
                    odd = True
    
    # create new object and add "text" attribute
    new_obj = {'text': result_string}
    # add new object to list
    new_dataset.append(new_obj)

# save the new dataset as a JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(new_dataset, output_file, indent=2)

print(f"New JSON file with 'text' attribute of length {len(new_dataset)} created at: {output_file_path}")
