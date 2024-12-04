import re

def get_unique_words(text):
    def remove_strings_with_digits(strings):
        # Use a list comprehension to filter out strings with any numeric character
        return [s for s in strings if not any(char.isdigit() for char in s)]
    
    # Normalize the text: convert to lowercase and remove non-alphanumeric characters
    words = re.findall(r'\b\w+\b', text.lower())  # \b matches word boundaries
    unique_words = set(words) # Use a set to eliminate duplicates
    return remove_strings_with_digits( list(unique_words) )

def save_list(l, output="list"):
    with open(output + ".txt",'w') as f:
        f.write(output + '\n')
        for word in l:
            f.write(word + '\n')

with open("wiki_italia.txt",'r') as text:
    a = text.read()
    l = get_unique_words(a)

save_list(l, "ITA_LARGE")
print(str(len(l)) + " parole trovate")

