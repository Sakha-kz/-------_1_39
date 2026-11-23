№1. CIFAR-10 + SmallCNN — эталонный ★ (разобран в образце)
Суть: CIFAR-10 32×32, SmallCNN с нуля; эталонный вариант, полностью
разобран в primer-vypolneniya-cv-variant-1.md. Вопрос: каков потолок маленькой
CNN с нуля и сколько даёт ImageNet-перенос? : сравнить с №2 (ResNet-18
scratch) на семинаре.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1001. --dataset cifar10 (10
классов, 32×32, 3 канала). Сплит: официальный train/test, train делим 90/10 на
train/val. EDA: гистограмма классов, contact sheet по 3 примера на класс. 2.
Классика (ст. 2). Лист обработки на 3 изображениях. Baseline: HOG +
LinearSVC — --classical hog. Сравни с CNN в таблице. 3. Аугментации (ст. 3).
Политика basic (flip+crop). Абляция: none vs basic, одинаковый seed. 4. Модель
(ст. 4). smallcnn (3 conv-блока, ~1.15M параметров). Посчитай params и память,
обоснуй выбор под 32×32. 5. Обучение (ст. 5). SGD lr 0.1, cosine, wd 5e-4, 30
эпох, batch 128. Кривые loss/acc, resume, best.pt. 6. Оценка (ст. 6). Полная:
accuracy/top-3/macro-F1, per-class P/R/F1, confusion matrix, error analysis (топ-5
пар + 10 ошибок), robustness. 7. Grad-CAM (ст. 7). 5 верных + 5 ошибочных;
смотрит ли модель на объект или на фон. 8. Перенос (ст. 8). transfer
ResNet-18: блок 9, probe (заморожен, 10 эп) → FT (backbone 1e-4 / head 1e-3, 10
эп), --norm imagenet --size 224. Таблица scratch vs probe vs FT. 9. Отчёт (ст. 9).
Latency, cost-модель, ответ на вопрос, таблица «классика vs scratch vs transfer».

Команды:

python train.py --dataset cifar10 --model smallcnn --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1001 --out_dir drive/out-var1
python evaluate.py --ckpt drive/out-var1/best.pt --dataset cifar10 --model smallcnn \
 --split val --seed 1001 --classical hog --processing --gradcam --robustness --latency --out_dir
drive/eval-var1

Артефакты: EDA + contact sheet, лист обработки, таблица
классика/CNN/transfer, кривые, confusion matrix, Grad-CAM, отчёт. Ловушка:
сравнивать классику и CNN на разных сплитах/seed — таблица сравнения
недействительна.

