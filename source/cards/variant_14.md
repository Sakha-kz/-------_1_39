№14. Flowers102 (20 классов) + ResNet-18
Суть: Flowers102 subset 20 классов, ResNet-18; фокус — fine-grained путаница
видов. Вопрос: топ-5 пар путаницы — чем похожи confundируемые виды? :
Grad-CAM на пары путаницы — на какие детали смотрит модель.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1014. flowers102, отбери 20
классов по seed, отфильтруй Wrapped.indices. EDA + contact sheet. 2. Классика
(ст. 2). Лист обработки. SIFT-BoW + LinearSVC: --classical siftbow. Различают ли
виды локальные особенности? 3. Аугментации (ст. 3). strong. Абляция none vs
strong. 4. Модель (ст. 4). resnet18 + ImageNet. Params/память. 5. Обучение (ст.
5). Transfer: probe→FT 10+10 (блок 9), --norm imagenet --size 224. Кривые, resume. 6.
Оценка (ст. 6). Полная + robustness; топ-5 пар путаницы из confusion matrix. 7.

Grad-CAM (ст. 7). 5+5; бонус — CAM на confundируемые пары (какие детали
различает модель). 8. Перенос (ст. 8). transfer + fine-grained FT: полный FT с
малым lr backbone; сравни probe vs FT на мелких различиях. 9. Отчёт (ст. 9).
Latency, cost, ответ на вопрос (чем похожи виды), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset flowers102 --model resnet18 --pretrained --aug strong --norm
imagenet \
 --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1014 --out_dir
drive/out-var14-ft
python evaluate.py --ckpt drive/out-var14-ft/best.pt --dataset flowers102 --model resnet18 \
 --norm imagenet --size 224 --split val --seed 1014 --classical siftbow --processing --gradcam --
robustness --latency --out_dir drive/eval-var14

Артефакты: топ-5 пар путаницы с примерами, Grad-CAM на пары, отчёт.
Ловушка: subset 20 классов должен совпадать с №15 (иначе сравнение CNN vs
ViT нечестное).

