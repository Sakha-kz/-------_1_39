№24. Caltech-101 (10 классов) + ResNet-18
Суть: Caltech-101 subset 10, ResNet-18; фокус — гетерогенные классы и
калибровка. Вопрос: какие классы «дороже» (самолёт vs краб vs пианино) и
почему? : калибровка уверенности (ECE) по классам.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1024. caltech101, отбери 10
классов по seed (офиц. test нет → 80/10/10). EDA + contact sheet. 2. Классика (ст.
2). Лист обработки. SIFT-BoW + LinearSVC: --classical siftbow. Воспроизвести до-
DL пайплайн Caltech. 3. Аугментации (ст. 3). basic. Абляция none vs basic. 4.
Модель (ст. 4). resnet18 + ImageNet. Params/память. 5. Обучение (ст. 5). Transfer
probe→FT 10+10 (блок 9), --norm imagenet --size 224. Кривые, resume. 6. Оценка
(ст. 6). Полная + robustness; ECE (expected_calibration_error) по классам. 7.
Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос (ст. 8). transfer; per-class
анализ гетерогенных классов: какие классы сложнее и почему. 9. Отчёт (ст.
9). Latency, cost, ответ на вопрос («дорогие» классы), ECE по классам (бонус),
таблица классика/scratch/transfer.

Команды:

python train.py --dataset caltech101 --model resnet18 --pretrained --aug basic --norm
imagenet \
 --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1024 --out_dir
drive/out-var24-ft
python evaluate.py --ckpt drive/out-var24-ft/best.pt --dataset caltech101 --model resnet18 \
 --norm imagenet --size 224 --split val --seed 1024 --classical siftbow --processing --gradcam --
robustness --latency --out_dir drive/eval-var24

Артефакты: per-class анализ, ECE по классам, таблица SIFT-BoW, отчёт.
Ловушка: Caltech-101 гетерогенен (разные размеры/фоны) — без per-class
анализа средний accuracy скрывает «дорогие» классы.

