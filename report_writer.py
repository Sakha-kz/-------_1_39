"""Academic reports based on measured files; missing results remain explicit."""
import json,re
from pathlib import Path
from cv39.registry import config

PROJECT=Path(__file__).resolve().parent
AUTHOR='### Ф.И.О. обучающегося\n### Сахтай Е.Н.\n### Шифр ОП\n### CSE8842\n### Ф.И.О. преподавателя\n### Ботайбеков А.О.'
STATES={'not_run':'Эксперимент не проведён','running':'Эксперимент выполняется','failed':'Выполнение прервано ошибкой','executed_requires_review':'Результаты получены; содержательный анализ не завершён'}
def read(p,default=None): return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
def fmt(v,d=4): return '—' if v is None else f'{v:.{d}f}' if isinstance(v,(int,float)) else str(v)
def save(p,parts): p.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')

def draw_curves(folder):
    histories=[(p.parent.relative_to(folder).as_posix(),read(p)) for p in folder.rglob('history.json')]
    if not histories: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name,history in histories:
        for ax,key in zip(axes,['val_accuracy','train_loss','lr']): ax.plot([r['epoch'] for r in history],[r[key] for r in history],label=name)
    for ax,title in zip(axes,['Validation accuracy','Training loss','Learning rate']): ax.set(xlabel='Epoch',title=title); ax.grid(alpha=.3)
    axes[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(folder/'curves.png',dpi=130); plt.close(fig)

def variant_report(folder,i):
    draw_curves(folder)
    cfg=read(folder/'config.json',config(i)); status=read(folder/'status.json',{'state':'not_run'}); split=read(folder/'split.json',{}); env=read(folder/'environment.json',{})
    card=(PROJECT/'source/cards'/f'variant_{i:02d}.md').read_text(encoding='utf-8')
    q=re.search(r'Вопрос:\s*(.*?)(?:\n\nЧто делать|Что делать по стадиям)',card,re.S)
    question=' '.join(q.group(1).split()) if q else 'Исследование качества классификации и особенностей выбранной архитектуры.'
    parts=[AUTHOR,f'# Отчёт по варианту {i}',f'Тема: {cfg["dataset"]} / {cfg["model"]}.',f'Состояние экспериментальной части: {STATES.get(status["state"],status["state"])}.',
        '## 1. Цель исследования',question,'## 2. Данные и воспроизводимость',
        f'Датасет: {cfg["dataset"]}. Seed обучения: {cfg["seed"]}; seed данных: {cfg["data_seed"]}.',
        '| Подвыборка | Изображений |\n|---|---:|',*[f'| {s} | {len(split[s]) if s in split else "—"} |' for s in ['train','val','test']],
        'Статистики нормализации и классические преобразования обучаются только на train. Validation используется для выбора модели, test выделен для финального измерения.',
        f'Устройство: {env.get("gpu") or ("CPU" if env else "не зафиксировано")}. PyTorch: {env.get("packages",{}).get("torch") or "не зафиксирован"}.',
        '## 3. Классические методы','Предусмотрены grayscale, Sobel, Canny, Otsu, opening/closing и skeletonization. Признаки данного варианта: '+', '.join(cfg['classical'])+'.',
        '| Метод | Accuracy | Macro-F1 | Признаков | Обучение, с |\n|---|---:|---:|---:|---:|']
    classics=read(folder/'classical/metrics.json',[])
    parts += [f'| {r["name"]} | {fmt(r["accuracy"])} | {fmt(r["macro_F1"])} | {r["feature_dim"]} | {fmt(r["train_seconds"],2)} |' for r in classics]
    if not classics: parts.append('| Измерения отсутствуют | — | — | — | — |')
    parts += ['## 4. Модель и обучение',f'Архитектура: {cfg["model"]}. Политика аугментации: {cfg["aug"]}. Базовый протокол: {cfg["optimizer"]}, LR {cfg["lr"]}, scheduler {cfg["scheduler"]}, batch {cfg["batch"]}; плановое число эпох {cfg["epochs"]}.',
        'Чекпоинты включают веса, optimizer, scheduler, AMP scaler, эпоху и состояния генераторов случайных чисел.',
        '| Прогон | Accuracy | Macro-F1 | Top-3 | Эпох | Параметров | Обучение, с |\n|---|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for p in sorted(folder.rglob('val_metrics.json')):
        m=read(p); t=read(p.parent/'training.json',{}); name=p.parent.relative_to(folder).as_posix(); rows.append((name,m,t,p.parent))
        parts.append(f'| {name} | {fmt(m["accuracy"])} | {fmt(m["macro_F1"])} | {fmt(m["top3"])} | {t.get("epochs_completed","—")} | {t.get("parameters","—")} | {fmt(t.get("train_seconds"),2)} |')
    if not rows: parts.append('| Полноразмерное обучение не выполнено | — | — | — | — | — | — |')
    a=next((r for r in rows if r[0]=='none'),None); b=next((r for r in rows if r[0]=='policy'),None)
    parts+=['## 5. Абляция аугментаций',f'Изменение validation accuracy policy−none: {b[1]["accuracy"]-a[1]["accuracy"]:+.6f}.' if a and b else 'Сравнение none и policy не завершено; изменение качества численно не установлено.',
        '## 6. Оценка качества и устойчивость']
    if b:
        parts+=['| Истинный класс | Предсказанный класс | Ошибок |\n|---|---|---:|']
        parts += [f'| {p["true"]} | {p["predicted"]} | {p["count"]} |' for p in b[1].get('top5_confusions',[])]
        corrupt=read(b[3]/'robustness.json',[])
        if corrupt:
            parts+=['| Искажение | Accuracy | Снижение относительно clean |\n|---|---:|---:|']
            parts += [f'| {r["corruption"]} | {fmt(r["accuracy"])} | {fmt(r["drop"])} |' for r in corrupt]
    else: parts.append('Полноразмерные confusion matrix, per-class показатели и результаты robustness отсутствуют.')
    parts+=['## 7. Интерпретируемость','Conv-модели исследуются с помощью Grad-CAM, ViT — attention rollout, MLP — градиентной saliency. Визуальные выводы относятся только к полученным картам.',
        '## 8. Специальный эксперимент',f'Предусмотренный сценарий: {cfg["experiment"]}.']
    special=list((folder/'special').rglob('*.json')) if (folder/'special').exists() else []
    parts.append('Сохранённые результаты специального эксперимента: '+', '.join(p.relative_to(folder).as_posix() for p in special)+'.' if special else 'Численные результаты специального эксперимента не получены.')
    for title,image in [('Распределение классов','class_histogram.png'),('Примеры классов','contact_sheet.png'),('Классическая обработка','processing.png'),('Кривые обучения','curves.png'),('Матрица ошибок','policy/confusion.png'),('Ошибочные примеры','policy/errors.png'),('Карты интерпретации','policy/explanations.png')]:
        if (folder/image).exists(): parts += [f'### {title}',f'![{title}]({image})']
    parts+=['## 9. Latency и экономика','| Прогон | Устройство | Latency, мс | FPS |\n|---|---|---:|---:|']
    rates_found=False
    for name,m,t,p in rows:
        for rate in read(p/'latency.json',[]):
            rates_found=True; parts.append(f'| {name} | {rate["device"]} | {fmt(rate["milliseconds"],3)} | {fmt(rate["FPS"],2)} |')
    if not rates_found: parts.append('| Измерения отсутствуют | — | — | — |')
    parts+=['Протокол latency: batch=1, 20 прогревочных прогонов, 100 измерений. Учебная цена T4 $0.2/час используется как допущение cost-модели; фактическим тарифом Colab она не является. Стоимость CPU без мощности и тарифа не рассчитана.',
        '## 10. Финальная оценка test']
    final=read(folder/'FINAL_TEST_LOCK.json',{})
    parts.append(f'Состояние: {final.get("state")}; accuracy: {fmt(final.get("accuracy"))}.' if final else 'Финальная оценка на test не проведена.')
    notes=folder/'REPORT_NOTES.md'; text=notes.read_text(encoding='utf-8').strip() if notes.exists() else ''
    if text.startswith(('Опишите','Заполните')): text=''
    parts+=['## 11. Выводы',text or ('Получены численные результаты. Содержательный анализ причин ошибок и ответ на индивидуальный вопрос ещё не зафиксированы.' if rows else 'Реализован экспериментальный протокол. Вывод о качестве и сравнении методов не сформирован, поскольку результаты полноразмерного обучения отсутствуют.')]
    save(folder/'README.md',parts); return status,len(rows)

def build_reports(root='runs'):
    root=Path(root); index=['# Сводные результаты 39 вариантов','| № | Датасет | Модель | Состояние |\n|---:|---|---|---|']; count=0
    for i in range(1,40):
        folder=root/'full'/f'variant_{i:02d}'; folder.mkdir(parents=True,exist_ok=True); state,n=variant_report(folder,i); count+=n; c=config(i)
        index.append(f'| [{i}](full/variant_{i:02d}/README.md) | {c["dataset"]} | {c["model"]} | {STATES.get(state["state"],state["state"])} |')
    save(root/'RESULTS.md',index)
    check=read(PROJECT/'verification/CHECKS.json',{}); smoke=[]
    for name in ['none','policy']:
        m=read(PROJECT/f'verification/runs/smoke/variant_29/{name}/val_metrics.json')
        if m: smoke.append(f'| {name} | {m["n"]} | {fmt(m["accuracy"])} | {fmt(m["macro_F1"])} |')
    parts=[AUTHOR,'# Отчёт по компьютерному зрению и дообучению Persian GPT-2','## Цель работы',
        'Разработка и исследование полного цикла классификации изображений для 39 вариантов: данные, классические признаки, нейросети, обучение, оценка, интерпретация и перенос. Отдельная часть посвящена адаптации Persian GPT-2 к поэтическому корпусу.',
        '## Реализованные методы','Предусмотрены HOG, LBP, HSV, Hu и SIFT-BoW с SVM/kNN; SmallCNN, LeNet, MLP, ResNet, MobileNet, DenseNet, EfficientNet и ViT. Специальные сценарии включают few-shot, дисбаланс, domain adaptation, дистилляцию, pruning/INT8 и SSL.',
        '## Воспроизводимость','Seed обучения определяется как 1000 + номер варианта; у сравниваемых пар зафиксирован общий seed данных. Чекпоинты включают состояние обучения и генераторов случайных чисел.',
        '## Результаты локальной проверки',f'Успешно выполнено модульных проверок: {check.get("tests_passed","не зафиксировано")}. Проверены resume, BatchNorm в linear probe, функции потерь, разбиение данных и маскирование padding.',
        'Проверочный эксперимент №29 выполнен на CPU: 120 обучающих и 48 validation-изображений, по две эпохи для none и policy.',
        '| Прогон | Validation, изображений | Accuracy | Macro-F1 |\n|---|---:|---:|---:|',*smoke,
        '## Полноразмерные эксперименты',f'Количество сохранённых наборов validation-метрик: {count}.',["[Сводная таблица всех вариантов](runs/RESULTS.md)"][0],
        '## Persian Poetry Fine-Tuning','Подготовлена адаптация базовой HooshvareLab/gpt2-fa. Полные стихотворения разделяются до токенизации; дубликаты удаляются. Предусмотрены held-out loss/perplexity и генерации до/после при одинаковых prompts и seed. Результаты полного fine-tuning в текущий комплект измерений не входят.',
        '## Ограничения исследования','Локальный проверочный запуск не позволяет судить о качестве 39 полноразмерных моделей. Дополнительные пункты вариантов отражены в [перечне покрытия](docs/COVERAGE.md). Сверка с исходным Poetry notebook ещё не подтверждена.',
        '## Выводы','Разработан экспериментальный проект; основные вычислительные компоненты проверены локально. Итоговые выводы о сравнении архитектур и качестве поэтической генерации зависят от полноразмерных измерений.',
        '## Сопроводительные материалы','[Отчёт Persian Poetry](poetry/README.md) · [Методические уточнения](docs/METHODOLOGY.md) · [Руководство запуска](docs/RUN_GUIDE.md)']
    save(root.parent/'README.md',parts)
