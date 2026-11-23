№2. CIFAR-10 + ResNet-18
Суть: CIFAR-10, ResNet-18 в двух режимах (scratch и transfer); фокус —
экономия от ImageNet-предобучения. Вопрос: сколько эпох и данных
экономит ImageNet-предобучение (кривые accuracy vs эпоха)? : сравнить
заморозку 0/50/100% backbone.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1002. cifar10. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. HOG vs raw pixels + LinearSVC: два
прогона (--classical hog и --classical raw), сравни, сколько даёт feature engineering.
3. Аугментации (ст. 3). basic. Абляция none vs basic. 4. Модель (ст. 4). resnet18.
Params/память; для transfer — веса ImageNet (--pretrained). 5. Обучение (ст. 5).
Scratch: SGD lr 0.1 cosine, 30 эпох. Transfer: probe→FT 10+10 (блок 9). Кривые,
resume. 6. Оценка (ст. 6). Полная + robustness; наложи кривые accuracy vs
эпоха для scratch и transfer. 7. Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос
(ст. 8). График разморозки: блок 9, постепенно unfreeze последние блоки
(layer4 → layer3 → …), замеряй val acc после каждого шага. 9. Отчёт (ст. 9).
Latency, cost, ответ на вопрос (экономия эпох/данных), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset cifar10 --model resnet18 --aug basic --optimizer sgd --lr 0.1 \
 --scheduler cosine --epochs 30 --seed 1002 --out_dir drive/out-var2-scratch
python evaluate.py --ckpt drive/out-var2-scratch/best.pt --dataset cifar10 --model resnet18 \
 --split val --seed 1002 --classical hog --processing --gradcam --robustness --latency --out_dir
drive/eval-var2

Артефакты: кривые scratch vs transfer, график разморозки, таблица HOG vs
raw, отчёт. Ловушка: для transfer забыть --norm imagenet --size 224 — качество
просядет, вывод об экономии будет неверным.

