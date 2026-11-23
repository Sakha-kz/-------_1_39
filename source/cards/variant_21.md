№21. Oxford Pets + DenseNet-121 transfer
Суть: Oxford Pets, DenseNet-121 transfer; фокус — Grad-CAM по породам
(морда/уши/окрас). Вопрос: какие визуальные признаки различают породы
(морда/уши/окрас)? : сравнить CAM с №20 — где лёгкая модель «смотрит не
туда».

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1021. oxford_pets. EDA +
contact sheet. 2. Классика (ст. 2). Лист обработки. SIFT-BoW + LinearSVC: --
classical siftbow. Что ловят SIFT-ключевые точки пород? 3. Аугментации (ст. 3).
basic. Абляция none vs basic. 4. Модель (ст. 4). densenet121 + ImageNet.
Params/память. 5. Обучение (ст. 5). Transfer probe→FT 10+10 (блок 9), --norm
imagenet --size 224. Кривые, resume. 6. Оценка (ст. 6). Полная + robustness. 7.
Grad-CAM (ст. 7). 5+5 по породам; какие детали (морда/уши/окрас) различает
модель. 8. Перенос (ст. 8). transfer + FT: полный протокол probe→FT; таблица
scratch vs probe vs FT. 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос (признаки
пород), сравнение CAM с №20 (бонус), таблица классика/scratch/transfer.

Команды:

python train.py --dataset oxford_pets --model densenet121 --pretrained --aug basic --norm
imagenet \
 --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1021 --out_dir
drive/out-var21-ft
python evaluate.py --ckpt drive/out-var21-ft/best.pt --dataset oxford_pets --model
densenet121 \
 --norm imagenet --size 224 --split val --seed 1021 --classical siftbow --processing --gradcam --
robustness --latency --out_dir drive/eval-var21

Артефакты: Grad-CAM по породам, сравнение CAM с №20, таблица SIFT-BoW,
отчёт. Ловушка: целевой слой Grad-CAM у DenseNet — последний conv блока
features, не classifier; проверь last_conv.

