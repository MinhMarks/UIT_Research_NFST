import json
import os
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
            
            # 1. Update columns list
            if 'columns = ["scaler","nCluster", \'noise_percentage\', "AUCROC"' in new_source:
                new_source = re.sub(
                    r'columns = \["scaler","nCluster", \'noise_percentage\', "AUCROC", "AUCPR", "Accuracy", "MCC", "F1 Score",\n\s+"Precision", "Recall", "Time Train", "Time Test"\]',
                    r'columns = ["scaler","nCluster", \'noise_percentage\', "AUCROC", "AUCPR", "Accuracy", "MCC", "F1 Score",\n           "Precision", "Recall", "Time Train", "Time Test", "Peak RAM Train (MB)"]',
                    new_source,
                    flags=re.DOTALL
                )

            # 2. Update result row saving logic in function()
            if 'def function(df1,df2, scaler, noise):' in new_source and 'v = Model_evaluating(y_test, y_predict, y_proba)' in new_source:
                new_source = re.sub(
                    r'(v = Model_evaluating\(y_test, y_predict, y_proba\)\n\s+# best_thresholds = BruteForce_Threshold\( y_test, y_proba, 0, 1\)\s+\n\s+)(result = \[scaler\] \+ \[ncluster\] \+ \[noise\] \+ v \+ \[training_time, inference_time\])',
                    r'import tracemalloc\n        current_train, peak_train = tracemalloc.get_traced_memory()\n        \1result = [scaler] + [ncluster] + [noise] + v + [training_time, inference_time, peak_train / 10**6]',
                    new_source,
                    flags=re.DOTALL
                )

            if source != new_source:
                lines = [line + '\n' if i < len(new_source.split('\n'))-1 else line 
                         for i, line in enumerate(new_source.split('\n'))]
                cell['source'] = lines
                modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f'Successfully updated output columns in {filepath}')
    else:
        print(f'No modification required or failed regex tracking in {filepath}')

except Exception as e:
    print(f"Error processing notebook: {e}")
