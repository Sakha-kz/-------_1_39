№11. SVHN + ResNet-20
Суть: SVHN (цифры на фото), ResNet-20 scratch; фокус — сколько ёмкости
съедают фон и цвет. Вопрос: сколько ёмкости съедают фон и цвет (сравнить с
№5)? : Grad-CAM — смотрит ли модель на цифру или на фон.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1011. svhn (label 0 = цифра 10
— учти при подписях). EDA + contact sheet. 2. Классика (ст. 2). Лист обработки.
colorhist (HSV) vs HOG + LinearSVC: два прогона (--classical colorhist, --classical
hog). Сколько даёт цвет? 3. Аугментации (ст. 3). Crop + ColorJitter, без flip
(цифры): правь basic в блоке 2 (убери flip, добавь T.ColorJitter). Абляция none vs

policy. 4. Модель (ст. 4). resnet20. Params/память. 5. Обучение (ст. 5). SGD lr 0.1
cosine, 30 эпох. Кривые, resume. 6. Оценка (ст. 6). Полная + robustness; сравни с
MNIST (№5). 7. Grad-CAM (ст. 7). 5+5; ключевой вопрос — цифра vs фон
(бонус). 8. Перенос (ст. 8). domain shift → MNIST: оцени на mnist, дообучи;
кривые forgetting/адаптации. 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос
(фон/цвет), таблица классика/scratch/transfer.

Команды:

python train.py --dataset svhn --model resnet20 --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1011 --out_dir drive/out-var11
python evaluate.py --ckpt drive/out-var11/best.pt --dataset svhn --model resnet20 \
 --split val --seed 1011 --classical colorhist --processing --gradcam --robustness --latency --
out_dir drive/eval-var11

Артефакты: таблица colorhist vs HOG, Grad-CAM цифра/фон, сравнение с
MNIST, отчёт. Ловушка: оставить flip в basic для SVHN — цифры искажаются;
и перепутать label 0 с цифрой 0 (это 10).

