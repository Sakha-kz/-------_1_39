№20. Oxford Pets (кошки/собаки) + MobileNetV3
Суть: Oxford Pets (37 пород), MobileNetV3; фокус — latency мобильной
архитектуры. Вопрос: accuracy vs скорость: что теряет лёгкая модель
(сравнить с №21)? : замер FPS на CPU и T4, таблица Pareto.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1020. oxford_pets (37 классов,
224×224). EDA + contact sheet. 2. Классика (ст. 2). Лист обработки. HOG +
LinearSVC: --classical hog. Latency классического пайплайна vs MobileNetV3. 3.
Аугментации (ст. 3). basic. Абляция none vs basic. 4. Модель (ст. 4). mobilenet_v3
+ ImageNet. Params/память; обоснуй мобильный выбор. 5. Обучение (ст. 5).
Transfer probe→FT 10+10 (блок 9), --norm imagenet --size 224. Кривые, resume. 6.
Оценка (ст. 6). Полная + robustness; обязательный latency (batch=1, CPU и T4).
7. Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос (ст. 8). few-shot по
породам: блок 3 few_shot_indices, 3 запуска mean±std; сравни с №21 (DenseNet).

9. Отчёт (ст. 9). Latency/FPS Pareto (бонус), cost, ответ на вопрос (что теряет
лёгкая модель), таблица классика/scratch/transfer.

Команды:

python train.py --dataset oxford_pets --model mobilenet_v3 --pretrained --aug basic --norm
imagenet \
 --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1020 --out_dir
drive/out-var20-ft
python evaluate.py --ckpt drive/out-var20-ft/best.pt --dataset oxford_pets --model
mobilenet_v3 \
 --norm imagenet --size 224 --split val --seed 1020 --classical hog --processing --gradcam --
robustness --latency --out_dir drive/eval-var20

Артефакты: таблица Pareto accuracy/FPS (CPU+T4), few-shot mean±std,
сравнение с №21, отчёт. Ловушка: latency без прогрева и без
torch.cuda.synchronize() — замеры шумные; следуй протоколу.

