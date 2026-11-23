"""Build portable Colab notebooks from this exact local source tree."""
import base64,gzip,json,re,uuid
from pathlib import Path
from cv39.registry import config

ROOT=Path(__file__).resolve().parent
def md(s): return dict(cell_type='markdown',metadata={},source=s.splitlines(True),id=uuid.uuid4().hex[:8])
def code(s): return dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[],id=uuid.uuid4().hex[:8])
def write(name,cells):
    obj=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={'colab':{'provenance':[]},'accelerator':'GPU',
        'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}})
    p=ROOT/'notebooks'/name; p.parent.mkdir(exist_ok=True,parents=True); p.write_text(json.dumps(obj,ensure_ascii=False,indent=1),encoding='utf-8')

def build():
    text=(ROOT/'source/39-cv-extracted.txt').read_text(encoding='utf-8')
    text=re.sub(r'<PARSED TEXT FOR PAGE: \d+ / \d+>','',text); matches=list(re.finditer(r'^№(\d+)\. ',text,re.M))
    cards={}
    for i,m in enumerate(matches):
        number=int(m.group(1))
        if 1<=number<=39: cards[number]=text[m.start():matches[i+1].start() if i+1<len(matches) else text.index('Глоссарий\n',m.start())]
    if len(cards)!=39: raise RuntimeError(f'Expected 39 complete cards, found {len(cards)}')
    folder=ROOT/'source/cards'; folder.mkdir(parents=True,exist_ok=True)
    coverage=['# Покрытие вариантов','Код подготовлен для запусков. Полное обучение ещё не выполнено. Бонусные и визуальные задания не считаются автоматически завершёнными.',
        '| № | Датасет | Модель | Спецэксперимент |\n|---:|---|---|---|']
    for i in range(1,40):
        (folder/f'variant_{i:02d}.md').write_text(cards[i],encoding='utf-8'); c=config(i)
        coverage.append(f'| {i} | {c["dataset"]} | {c["model"]} | {c["experiment"]} |')
    coverage+=['## Дополнительные условия, которые требуют проверки',
        '- №4: внутри-суперклассовый анализ требует отдельной fine-головы; текущая модель coarse-классификатор.',
        '- №5: ансамбль из трёх моделей — бонус, не включён в базовый автоматический сценарий.',
        '- №6: выравниваются параметры MLP и LeNet; картинки весов первого слоя/HOG требуют отдельного добавления.',
        '- №8: probe по layer2/3/4 — бонус, текущий probe использует стандартную последнюю голову.',
        '- №13: pseudo-label STL-10 unlabeled — бонус, текущий few-shot использует только размеченный train.',
        '- №17/19: классы аэродатасетов не совпадают; перенос признаков с новой головой, а не фиктивный zero-shot.',
        '- №16: адаптация DTD на трёх синтетических текстурах и GAP/flatten реализованы; требуется визуальный анализ.',
        '- №20: CPU+GPU latency измеряется; few-shot k=10 выбран явно, поскольку k в карточке не задан.',
        '- №24: общая ECE есть, дополнительная class-wise calibration требует анализа отдельно.',
        '- №26: coverage/accuracy сохраняется; перечень опасных пар выбирается по смыслу знаков вручную.',
        '- №27: обязательное сравнение с №26 возможно после обоих запусков; INT8-бонус отдельно от №37 не запускается.',
        '- №29/30: новый цвет и shift/scale реализованы; heatmap позиции/rotation — дополнительные измерения.',
        '- №32: два LR на оптимизатор; Lion/LAMB — бонус, не включён.',
        '- №33: три scheduler реализованы; warmup-абляция — бонус.',
        '- №34: четыре регуляризатора и SVM C-sweep реализованы; сочетание двух лучших — бонус.',
        '- №35: weighted, sampling, focal реализованы; DRW — бонус.',
        '- №37: физический pruning/INT8 четыре конфигурации; recovery fine-tune — дополнительный бонус.',
        '- №39: общий бюджет может не позволить все компоненты; неполнота отражается в budget.json.',
        '- Всем: визуальные причины ошибок, индивидуальный вопрос и межвариантные сравнения должны опираться на полученные файлы.']
    (ROOT/'docs/COVERAGE.md').write_text('\n\n'.join(coverage),encoding='utf-8')
    files={}
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if p.is_file() and not any(x in rel.parts for x in ['notebooks','runs','tests','__pycache__','verification']) and p.suffix in ['.py','.md','.txt']:
            files[rel.as_posix()]=p.read_text(encoding='utf-8')
    payload=base64.b64encode(gzip.compress(json.dumps(files,ensure_ascii=False).encode())).decode()
    bootstrap='''import base64, gzip, json, os, sys
from pathlib import Path
PROJECT=Path('/content/cv39_project')
PROJECT.mkdir(exist_ok=True)
payload = '''+repr(payload)+'''
files=json.loads(gzip.decompress(base64.b64decode(payload)))
for relative, content in files.items():
    p=PROJECT/relative
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(content,encoding='utf-8')
os.chdir(PROJECT)
sys.path.insert(0,str(PROJECT))
print('Source ready:',PROJECT)
'''
    drive="from google.colab import drive\ndrive.mount('/content/drive')\nOUT='/content/drive/MyDrive/CV39_Sakhtay/runs'\nDATA='/content/data'\nPath(DATA).mkdir(exist_ok=True)\n"
    install='%pip install -q -r requirements.txt\n'
    preflight="import torch\nfrom cv39.util import environment\nprint(environment())\nassert torch.cuda.is_available(), 'Select T4 GPU in Runtime settings'\n"
    common=[md('## Исходники проекта\nЭта ячейка восстанавливает код из пакета. Работа выполняется в Colab, файлы результатов — на вашем Drive.'),code(bootstrap),code(install),code(drive),code(preflight)]
    write('00_Start_Here.ipynb',[md('# Все 39 вариантов — запуск в Colab\nСахтай Е.Н. · CSE8842 · Ботайбеков А.О.\n\nСначала включите T4 GPU. Откройте README. Проверочный вариант 29 не скачивает данные. Затем запускайте полноразмерные эксперименты по одному. Это не готовые результаты обучения.')]+common+[
        md('## Проверка пайплайна\nДва коротких прогона synthetic. Не использовать smoke-метрики как результат задания.'),
        code("import subprocess\nsubprocess.run([sys.executable,'run.py','--variant','29','--smoke','--stage','train','--out',OUT,'--data',DATA,'--workers','0'],check=True)"),
        md('## Очередь полноразмерных запусков\nИзмените список номеров. Не запускайте повторно уже завершённые варианты без причины. Сохраните notebook с выводами.'),
        code("VARIANTS=[1]\nfor number in VARIANTS:\n    subprocess.run([sys.executable,'run.py','--variant',str(number),'--out',OUT,'--data',DATA],check=True)"),
        md('## Сборка отчётов'),code("from report import build_reports\nbuild_reports(OUT)\nprint(Path(OUT,'RESULTS.md').read_text(encoding='utf-8'))")])
    for i in range(1,40):
        c=config(i); cells=[md(f'# Вариант {i} — {c["dataset"]} / {c["model"]}\nСахтай Е.Н. · CSE8842 · Ботайбеков А.О.\n\n**Модельный эксперимент ещё не выполнен.** Оригинальная карточка включена ниже.\n\n'+cards[i])]+common+[
            md('## Параметры\nSMOKE=True проверяет только короткое обучение. Для выполнения задания используйте False. UC Merced, QuickDraw и UTKFace предварительно разместите по README.'),
            code(f"VARIANT={i}\nSMOKE=False\n# DATA можно заменить на папку с подготовленными данными на Drive.\n"),
            md('## Обучение, оценка и специальные эксперименты\nПосле обрыва повторите эту же ячейку. Resume допускает только прежнюю конфигурацию.'),
            code("import subprocess\ncmd=[sys.executable,'run.py','--variant',str(VARIANT),'--out',OUT,'--data',DATA]\nif SMOKE: cmd+=['--smoke','--stage','train','--workers','0']\nsubprocess.run(cmd,check=True)"),
            md('## Отчёт по полученным результатам'),
            code("from report import build_reports\nfrom IPython.display import display, Markdown\nbuild_reports(OUT)\nfolder=Path(OUT)/('smoke' if SMOKE else 'full')/f'variant_{VARIANT:02d}'\nif (folder/'README.md').exists(): display(Markdown((folder/'README.md').read_text(encoding='utf-8')))\nelse: print('Smoke проверяет код; основной отчёт находится в runs/full')"),
            md('## Ручной анализ\nПроверьте изображения, классы, причины ошибок и ответ на индивидуальный вопрос. Создайте REPORT_NOTES.md рядом с результатами. Открывайте docs/COVERAGE.md: автоматический сценарий не заменяет бонусы и интерпретацию.'),
            code("notes=folder/'REPORT_NOTES.md'\nif not notes.exists(): notes.write_text('Опишите реальные наблюдения и ответ на вопрос карточки.\\n',encoding='utf-8')\nprint(notes)"),
            md('## Финальный test — один раз после выбора по val\nПо умолчанию выключен. Проверьте, что выбран исходный классификатор, а не другая domain/rotation модель.'),
            code("FINAL_TEST=False\nif FINAL_TEST and not SMOKE:\n    subprocess.run([sys.executable,'run.py','--variant',str(VARIANT),'--stage','test','--out',OUT,'--data',DATA],check=True)"),
            md('## Перед сдачей\nСохраните исполненный .ipynb через меню Colab, загрузите README/JSON/CSV/рисунки в GitHub. Веса и данные оставьте на Drive.')]
        write(f'CV_{i:02d}.ipynb',cells)
    poetry=[md('# Persian Poetry Fine-Tuning\nСначала сохраняем оригинал задания, затем запускаем современную адаптацию. Она не выдаётся за точную копию непроверенного оригинала. GPU-результаты ещё не получены.')]+common+[
        md('## Оригинальный notebook и ревизия'),code("from poetry.fetch_original import fetch\noriginal=fetch()\nprint(Path('poetry/original_code.py.txt').read_text(encoding='utf-8'))\n# Сохраните оригинальный .ipynb и сравните его ячейки с нашей версией.\nfrom google.colab import files\nfiles.download(original)"),
        md('## Открытый корпус стихов\nИсточник: aghasemi/ChronologicalPersianPoetryDataset, CC-BY-SA-4.0. Проверьте фактические колонки TSV. Не смешивайте строки одного стихотворения между train/val.'),
        code("import subprocess,csv\ncorpus=Path('/content/persian_poetry_data')\nif not corpus.exists(): subprocess.run(['git','clone','--depth','1','https://github.com/aghasemi/ChronologicalPersianPoetryDataset.git',str(corpus)],check=True)\nTSV=corpus/'poems.tsv'\nwith TSV.open(encoding='utf-8-sig',newline='') as f:\n    reader=csv.DictReader(f,delimiter='\\t')\n    print('Columns:',reader.fieldnames)\n    print(next(reader))\nprint('Dataset revision:',subprocess.check_output(['git','-C',str(corpus),'rev-parse','HEAD'],text=True).strip())"),
        md('## Укажите колонки из вывода выше\nВыберите поле полного стихотворения и, если доступно, автора. По умолчанию колонка не угадывается.'),
        code("TEXT_COLUMN=''\nPOET_COLUMN=''\nLIMIT=5000\nEPOCHS=1\nBLOCK_SIZE=128\nPOETRY_OUT='/content/drive/MyDrive/CV39_Sakhtay/poetry'\nassert TEXT_COLUMN, 'Укажите реальное имя текстовой колонки из предыдущей ячейки'"),
        md('## Дообучение базовой GPT-2 и сравнение до/после'),
        code("cmd=[sys.executable,'poetry/train_poetry.py','--tsv',str(TSV),'--text-column',TEXT_COLUMN,'--out',POETRY_OUT,'--limit',str(LIMIT),'--epochs',str(EPOCHS),'--block',str(BLOCK_SIZE)]\nif POET_COLUMN: cmd+=['--poet-column',POET_COLUMN]\nsubprocess.run(cmd,check=True)"),
        md('## Фактические результаты'),code("for filename in ['before.json','after.json']:\n    print(filename)\n    print(Path(POETRY_OUT,filename).read_text(encoding='utf-8'))"),
        md('## Отчёт\nЗапишите версии и ревизии, размер подвыборки, baseline и after perplexity, одинаковые prompts/seed и свои наблюдения о связности, повторениях и стиле. Сверьте формат токенов и гиперпараметры с оригиналом и перечислите отличия. Сохраните исполненный notebook.')]
    write('Poetry_Colab.ipynb',poetry)
    print('Built 41 notebooks and 39 original variant cards.')

if __name__=='__main__': build()
