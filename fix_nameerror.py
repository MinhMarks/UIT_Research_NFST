import json
import re

filepath = 'notebooks/experiments/OC-NSFT_old_noise_Kmean_threshold.ipynb'

try:
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            new_source = source
            
            if 'def calculate_NPD' in new_source:
                # Remove the improperly injected block
                new_source = re.sub(
                    r'\s+current_mem, peak_mem = tracemalloc\.get_traced_memory\(\)\n\s+tracemalloc\.stop\(\)\n\s+print\(f"--- Peak Memory for Learn API: {peak_mem / 10\*\*6:\.2f} MB ---"\)\n\s+print\(f"Size of null_point_X: {sys\.getsizeof\(null_point_X\) / 10\*\*6:\.2f} MB"\)\n\s+print\(f"Size of null_point_X_test: {sys\.getsizeof\(null_point_X_test\) / 10\*\*6:\.2f} MB"\)',
                    '',
                    new_source
                )

            if source != new_source:
                lines = [line + '\n' if i < len(new_source.split('\n'))-1 else line 
                         for i, line in enumerate(new_source.split('\n'))]
                cell['source'] = lines
                modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f'Successfully fixed NameError in {filepath}')
    else:
        print(f'No matching regex to fix in {filepath}')

except Exception as e:
    print(f"Error processing notebook: {e}")
