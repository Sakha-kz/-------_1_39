№37. CIFAR-10 + прунинг и квантизация ★
Суть: CIFAR-10, ResNet-18; фокус — accuracy vs latency vs размер (deployment).
Вопрос: что дешевле для deployment: прунинг или квантизация, и можно ли
совмещать? : fine-tune после прунинга — восстановление качества.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1037. cifar10. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. HOG + LinearSVC: --classical hog.
Deployment-эталон: размер/latency классики. 3. Аугментации (ст. 3). basic.
Абляция none vs basic. 4. Модель (ст. 4). resnet18 (обучить).
Params/память/размер файла. 5. Обучение (ст. 5). SGD lr 0.1 cosine, 30 эпох.
Кривые, resume. 6. Оценка (ст. 6). Полная + robustness; таблица Pareto
accuracy/latency/размер. 7. Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос (ст.
8). Прунинг 50% каналов + INT8-квантизация (по отдельности и вместе, блок
11 — отправные точки, реализация твоя); fine-tune после прунинга (бонус). 9.
Отчёт (ст. 9). Latency/размер до/после, cost, ответ на вопрос (что дешевле),
таблица классика/scratch/pruned/quant.

Команды:

python train.py --dataset cifar10 --model resnet18 --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1037 --out_dir drive/out-var37
python evaluate.py --ckpt drive/out-var37/best.pt --dataset cifar10 --model resnet18 \
 --split val --seed 1037 --classical hog --processing --gradcam --robustness --latency --out_dir
drive/eval-var37

Артефакты: таблица Pareto (4 конфигурации), код прунинга/квантизации,
отчёт. Ловушка: прунинг без fine-tune резко роняет acc — замеряй и
восстановление после дообучения.

