№13. STL-10 + ResNet-18 transfer
Суть: STL-10 (96×96), ResNet-18 transfer; фокус — transfer при малых данных.
Вопрос: сколько примеров нужно для 90% качества полного обучения
(transfer + мало данных)? : использовать unlabeled-сплит STL-10 (pseudo-
labels).

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1013. stl10 (10 классов, 96×96;
есть +100k unlabeled). EDA + contact sheet. 2. Классика (ст. 2). Лист обработки.
SIFT-BoW + LinearSVC: --classical siftbow. Transfer vs классика при малых
данных. 3. Аугментации (ст. 3). basic. Абляция none vs basic. 4. Модель (ст. 4).
resnet18 + ImageNet (--pretrained). Params/память. 5. Обучение (ст. 5). Transfer:
probe→FT 10+10 (блок 9), --norm imagenet --size 224. Кривые, resume. 6. Оценка
(ст. 6). Полная + robustness. 7. Grad-CAM (ст. 7). 5+5; объект vs фон. 8. Перенос
(ст. 8). few-shot k=50 на класс: блок 3 few_shot_indices(targets, 50, seed), 3 запуска
mean±std; кривая «качество vs число примеров», найди точку 90%. 9. Отчёт
(ст. 9). Latency, cost, ответ на вопрос (сколько примеров для 90%), таблица
классика/scratch/transfer/few-shot.

Команды:

python train.py --dataset stl10 --model resnet18 --pretrained --aug basic --norm imagenet \
 --size 224 --optimizer sgd --lr 1e-3 --scheduler cosine --epochs 10 --seed 1013 --out_dir
drive/out-var13-ft
python evaluate.py --ckpt drive/out-var13-ft/best.pt --dataset stl10 --model resnet18 \
 --norm imagenet --size 224 --split val --seed 1013 --classical siftbow --processing --gradcam --
robustness --latency --out_dir drive/eval-var13

Артефакты: кривая качество vs k, few-shot mean±std, pseudo-labels (бонус),
отчёт. Ловушка: few-shot без аугментаций и без 3 запусков — разброс не
оценить, протокол не выполнен.

