import os
import json
import xml.etree.ElementTree as ET


def extract_data_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    print(root.findall('INDIVIDUAL'))
    for individual in root.findall('INDIVIDUAL'):
        entry = {}

        for child in individual:

            if child.tag == 'WRITING':

                writing_info = {}
                for writing_child in child:
                    writing_info[writing_child.tag] = writing_child.text.strip()
                entry['WRITING'] = writing_info
            else:
                entry[child.tag] = child.text.strip()
        data.append(entry)

    return data


def process_xml_files(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            xml_file = os.path.join(directory, filename)
            data = extract_data_from_xml(xml_file)

            all_data.extend(data)
    print (all_data)
    return all_data


def save_data_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    data_directory1 = './data/2017_cases/2017_cases/neg/'
    data_directory2 = './data/2017_cases/2017_cases/pos/'
    output_file = './data/train_new_data.json'
    print(len(process_xml_files(data_directory1)),len(process_xml_files(data_directory2)))
    all_data = process_xml_files(data_directory1)+process_xml_files(data_directory2)
    save_data_to_json(all_data, output_file)