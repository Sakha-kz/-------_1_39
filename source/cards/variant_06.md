№6. MNIST + MLP ★ (CPU-friendly)
Суть: MNIST, MLP (2 скрытых слоя) scratch; фокус — нужны ли свёртки для
MNIST. Вопрос: нужны ли свёртки для MNIST — MLP vs LeNet (№5) при равных
params? : визуализировать выученные веса первого слоя (что выучил MLP).

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1006. mnist. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. raw pixels + LinearSVC: --classical raw.
Визуализируй HOG цифр (что «видит» дескриптор). 3. Аугментации (ст. 3).
none vs noise (гауссов шум σ=0.05). Абляция обязательна. 4. Модель (ст. 4). mlp
(2 скрытых слоя). Подбери ширину так, чтобы params ≈ LeNet №5; зафиксируй
факт. 5. Обучение (ст. 5). SGD lr 0.1 cosine, 15 эпох. Кривые, resume. 6. Оценка
(ст. 6). Полная + robustness; сравни с LeNet (№5) при равных params. 7. Grad-
CAM (ст. 7). Для MLP — saliency/визуализация весов первого слоя (бонус); 5+5
ошибок. 8. Перенос (ст. 8). few-shot k=10 на класс: блок 3
few_shot_indices(targets, 10, seed), 3 запуска mean±std; сравни с полным
обучением. 9. Отчёт (ст. 9). Latency (CPU), cost, ответ на вопрос (нужны ли
свёртки), таблица классика/scratch/few-shot.

Команды:

python train.py --dataset mnist --model mlp --aug noise --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 15 --seed 1006 --out_dir drive/out-var6
python evaluate.py --ckpt drive/out-var6/best.pt --dataset mnist --model mlp \
 --split val --seed 1006 --classical raw --processing --robustness --latency --out_dir drive/eval-
var6

Артефакты: визуализация весов 1-го слоя, таблица MLP vs LeNet, few-shot
mean±std, отчёт. Ловушка: сравнивать MLP и LeNet без выравнивания params
— вывод о свёртках будет некорректным.

