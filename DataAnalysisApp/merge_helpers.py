from pathlib import Path

app_path = Path('app.py')
utils_path = Path('utils.py')
file_handler_path = Path('file_handler.py')
stats_tests_path = Path('stats_tests.py')
preprocessing_path = Path('preprocessing.py')
ml_models_path = Path('ml_models.py')
deep_learning_path = Path('deep_learning.py')

text = app_path.read_text(encoding='utf-8')
start_marker = 'from utils'
end_marker = '# ════════════════════════════════════════════════════════════════'
old_start = text.index(start_marker)
old_end = text.index(end_marker, old_start)

combined = []
combined.append(end_marker)
combined.append('# INLINED HELPER MODULES')
combined.append(end_marker)
combined.append(utils_path.read_text(encoding='utf-8'))
combined.append(file_handler_path.read_text(encoding='utf-8'))
combined.append(stats_tests_path.read_text(encoding='utf-8'))
combined.append(preprocessing_path.read_text(encoding='utf-8'))
combined.append(ml_models_path.read_text(encoding='utf-8'))
combined.append(deep_learning_path.read_text(encoding='utf-8'))
new_block = '\n'.join(combined) + '\n\n'

new_text = text[:old_start] + new_block + text[old_end:]
app_path.write_text(new_text, encoding='utf-8')
print('Merged helper modules into app.py')
