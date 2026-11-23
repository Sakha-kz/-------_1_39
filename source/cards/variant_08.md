№8. Fashion-MNIST + ResNet-18 transfer
Суть: Fashion-MNIST, ResNet-18 transfer; фокус — эффективность linear probe.
Вопрос: сколько % качества FT даёт probe при ~1% обучаемых параметров? :
probe на признаках разных глубин (layer2/3/4).

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1008. fashion_mnist. EDA +
contact sheet. 2. Классика (ст. 2). Лист обработки. LBP + LinearSVC: --classical lbp.

Per-class F1 текстурного дескриптора. 3. Аугментации (ст. 3). basic. Абляция
none vs basic. 4. Модель (ст. 4). resnet18 + ImageNet (--pretrained). Grayscale→3
канала: повтори канал x.repeat(1,3,1,1) или адаптируй первый conv (блок 5).
Params/память. 5. Обучение (ст. 5). Transfer: probe (10 эп) → FT (10 эп), блок 9, --
norm imagenet --size 224. Кривые, resume. 6. Оценка (ст. 6). Полная + robustness;
посчитай долю обучаемых params в probe vs FT. 7. Grad-CAM (ст. 7). 5+5;
объект vs фон. 8. Перенос (ст. 8). linear probe vs full FT (полный протокол):
таблица scratch vs probe vs FT (acc, время, обучаемые params). Бонус — probe
на layer2/3/4. 9. Отчёт (ст. 9). Latency, cost, ответ на вопрос (эффективность
probe), таблица классика/scratch/probe/FT.

Команды:

python train.py --dataset fashion_mnist --model resnet18 --pretrained --aug basic --norm
imagenet \
 --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1008 --out_dir
drive/out-var8-ft
python evaluate.py --ckpt drive/out-var8-ft/best.pt --dataset fashion_mnist --model resnet18 \
 --norm imagenet --size 224 --split val --seed 1008 --classical lbp --processing --gradcam --
robustness --latency --out_dir drive/eval-var8

Артефакты: таблица probe vs FT (params/acc), probe по глубинам (бонус),
отчёт. Ловушка: probe без заморозки BN (переведи backbone в eval) —
статистики BN «протекают», probe перестаёт быть честным.

