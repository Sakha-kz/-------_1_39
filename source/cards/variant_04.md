№4. CIFAR-100 (10 суперклассов) + ResNet-20
Суть: CIFAR-100, subset из 10 суперклассов, ResNet-20 scratch; фокус — рост
сложности с числом классов. Вопрос: как растёт сложность задачи с числом
классов (сравнить с №1)? : ошибка «внутри суперкласса» vs «между
суперклассами» — разбор матрицы.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1004. cifar100, отбери 10
суперклассов (coarse labels) по seed, отфильтруй Wrapped.indices. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. HOG + kNN (k=5): код блока 4,
clf='knn'. Метрический классификатор на 10 суперклассах. 3. Аугментации (ст.
3). basic. Абляция none vs basic. 4. Модель (ст. 4). resnet20 (ResNetCIFAR, в
torchvision готового нет). Params/память. 5. Обучение (ст. 5). SGD lr 0.1 cosine,
30 эпох. Кривые, resume. 6. Оценка (ст. 6). Полная + robustness; confusion
matrix по суперклассам. 7. Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос (ст.
8). transfer + иерархия: ResNet-18 ImageNet probe→FT; сравни ошибку внутри
суперкласса vs между суперклассами. 9. Отчёт (ст. 9). Latency, cost, ответ на
вопрос (сравнение с №1), таблица классика/scratch/transfer.

Команды:

python train.py --dataset cifar100 --model resnet20 --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1004 --out_dir drive/out-var4
python evaluate.py --ckpt drive/out-var4/best.pt --dataset cifar100 --model resnet20 \
 --split val --seed 1004 --processing --gradcam --robustness --latency --out_dir drive/eval-var4

Артефакты: subset-скрипт, confusion matrix, разбор иерархии ошибок,
таблица HOG+kNN, отчёт. Ловушка: классику (kNN) считать через --classical —
флаг не умеет kNN; нужен код блока 4.

