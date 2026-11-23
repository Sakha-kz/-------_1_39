"""Reports use measured files; absence stays explicit rather than becoming zero."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cv39.registry import config

def read(p,default=None): return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default

def build_reports(root='runs'):
    root=Path(root); index=['# Результаты всех 39 вариантов','| № | Датасет | Модель | Статус |\n|---:|---|---|---|']
    for i in range(1,40):
        c=config(i); folder=root/'full'/f'variant_{i:02d}'; folder.mkdir(parents=True,exist_ok=True)
        status=read(folder/'status.json',{'state':'not_run'}); index.append(f'| {i} | {c["dataset"]} | {c["model"]} | {status["state"]} |')
        rows=[]; histories=[]
        for path in sorted(folder.rglob('val_metrics.json')):
            m=read(path); training=read(path.parent/'training.json',{}); latency=read(path.parent/'latency.json',[{}])[0]
            name=path.parent.relative_to(folder).as_posix(); seconds=training.get('train_seconds')
            rows.append(dict(run=name,accuracy=m['accuracy'],macro_F1=m['macro_F1'],top3=m['top3'],parameters=training.get('parameters'),
                             train_seconds=seconds,training_cost_usd=seconds/3600*.2 if seconds is not None else None,
                             inference_1000_cost_usd=latency.get('milliseconds',0)*1000/1000/3600*.2 if latency.get('milliseconds') is not None else None))
            history=read(path.parent/'history.json',[])
            if history: histories.append((name,history))
        if histories:
            fig,axes=plt.subplots(1,3,figsize=(15,4))
            for name,h in histories:
                ep=[r['epoch'] for r in h]
                axes[0].plot(ep,[r['val_accuracy'] for r in h],label=name); axes[1].plot(ep,[r['train_loss'] for r in h],label=name); axes[2].plot(ep,[r['lr'] for r in h],label=name)
            for ax,title in zip(axes,['Val accuracy','Train loss','Learning rate']): ax.set(xlabel='Epoch',title=title); ax.grid(alpha=.3)
            axes[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(folder/'curves.png',dpi=130); plt.close(fig)
        lines=['### Ф.И.О. обучающегося\n### Сахтай Е.Н.\n### Шифр ОП\n### CSE8842\n### Ф.И.О. преподавателя\n### Ботайбеков А.О.',
               f'# Вариант {i}: {c["dataset"]} / {c["model"]}',f'**Статус: {status["state"]}.**',
               f'Seed обучения: {c["seed"]}; seed данных: {c["data_seed"]}. Индивидуальная работа.',
               '## Цель и задание\nПолная карточка варианта: `source/cards/variant_'+f'{i:02d}.md`.',
               '## Данные и воспроизводимость\nСм. config.json, environment.json и split.json. Статистики нормализации вычислены только на train.',
               '## Сравнение измеренных результатов',
               '| Прогон | Accuracy | Macro-F1 | Top-3 | Train, с | Параметры |\n|---|---:|---:|---:|---:|---:|']
        for r in rows: lines.append(f'| {r["run"]} | {r["accuracy"]:.4f} | {r["macro_F1"]:.4f} | {r["top3"]:.4f} | {r["train_seconds"]} | {r["parameters"]} |')
        if not rows: lines.append('| Обучение не выполнено | — | — | — | — | — |')
        classics=read(folder/'classical/metrics.json',[])
        for r in classics: lines.append(f'| Классика {r["name"]} | {r["accuracy"]:.4f} | {r["macro_F1"]:.4f} | — | {r["train_seconds"]:.2f} | признаков {r["feature_dim"]} |')
        none=next((r for r in rows if r['run']=='none'),None); policy=next((r for r in rows if r['run']=='policy'),None)
        lines+=['## Абляция аугментаций',f'Δaccuracy policy−none: {policy["accuracy"]-none["accuracy"]:.6f}' if none and policy else 'Результаты обоих прогонов ещё не получены.']
        for title,img in [('EDA','class_histogram.png'),('Примеры классов','contact_sheet.png'),('Классическая обработка','processing.png'),('Кривые обучения','curves.png'),('Матрица ошибок','policy/confusion.png'),('Ошибки','policy/errors.png'),('Интерпретируемость','policy/explanations.png')]:
            if (folder/img).exists(): lines += [f'## {title}',f'![{title}]({img})']
        lines+=['## Специальный эксперимент\nСм. JSON и изображения в special/. Файлы создаются после соответствующих запусков.',
                '## Latency и экономика\nLatency: batch=1, 20 warmup, 100 измерений (или меньше только в smoke). '+
                'Цена $0.2/GPU-час — учебное допущение из задания, не текущий тариф Colab и не фактический счёт. CPU-стоимость без мощности и тарифа не вычисляется.',
                '```json\n'+json.dumps(rows,ensure_ascii=False,indent=2)+'\n```',
                '## Анализ и ответ на вопрос',
                (folder/'REPORT_NOTES.md').read_text(encoding='utf-8') if (folder/'REPORT_NOTES.md').exists() else 'Требуется визуальный разбор результатов. Причины ошибок и качество генерализации не определяются автоматически.',
                '## Финальный test\n'+json.dumps(read(folder/'FINAL_TEST_LOCK.json',{'state':'not_evaluated'}),ensure_ascii=False)]
        (folder/'README.md').write_text('\n\n'.join(lines),encoding='utf-8')
    (root/'RESULTS.md').write_text('\n\n'.join(index),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='runs'); a=p.parse_args(); build_reports(a.out)
