№32. CIFAR-10 + sweep оптимизаторов
Суть: CIFAR-10, ResNet-18, 3 оптимизатора; фокус — SGD+momentum vs
Adam/AdamW. Вопрос: почему SGD+momentum часто лучше Adam на CNN
(обобщение vs скорость)? : добавить LAMB/Lion (timm) — четвёртый прогон.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1032. cifar10. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. HOG + LinearSVC: --classical hog.
Точка отсчёта: классика без оптимизации. 3. Аугментации (ст. 3). basic.
Абляция none vs basic (для лучшего оптимизатора). 4. Модель (ст. 4). resnet18.
Params/память. 5. Обучение (ст. 5). 3 прогона: SGD+momentum (lr 0.1) / Adam

(lr 1e-3) / AdamW (lr 1e-3), cosine, 30 эпох. Подбери lr каждому. Кривые, resume.
6. Оценка (ст. 6). Полная + robustness; таблица acc/скорость сходимости по
оптимизаторам. 7. Grad-CAM (ст. 7). 5+5 для лучшего; объект vs фон. 8.
Перенос (ст. 8). Лучший оптимизатор → FT: transfer probe→FT (блок 9). 9.
Отчёт (ст. 9). Latency, cost, ответ на вопрос (обобщение vs скорость), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset cifar10 --model resnet18 --aug basic --optimizer adamw --lr 1e-3 \
 --scheduler cosine --epochs 30 --seed 1032 --out_dir drive/out-var32-adamw
python evaluate.py --ckpt drive/out-var32-adamw/best.pt --dataset cifar10 --model resnet18 \
 --split val --seed 1032 --classical hog --processing --gradcam --robustness --latency --out_dir
drive/eval-var32

Артефакты: таблица 3 оптимизаторов (acc + кривые), подбор lr, отчёт.
Ловушка: сравнивать оптимизаторы с одним lr без подбора — Adam с lr 0.1
разойдётся; каждому свой lr.

