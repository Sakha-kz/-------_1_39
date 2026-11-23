### Ф.И.О. обучающегося
### Сахтай Е.Н.
### Шифр ОП
### CSE8842
### Ф.И.О. преподавателя
### Ботайбеков А.О.

# Компьютерное зрение — 39 вариантов и Persian Poetry Fine-Tuning

Репозиторий содержит код для всех 39 вариантов из `39-cv.doc`, Colab-ноутбуки,
сборку отчётов из сохранённых метрик и отдельную адаптацию Persian GPT-2.


## Начать с Google Colab

1. Скачайте и распакуйте пакет. В Colab выберите **Файл → Загрузить блокнот**.
2. Откройте `notebooks/00_Start_Here.ipynb`, включите **Среда выполнения →
   Сменить среду → T4 GPU** и выполните ячейки проверки.
3. Для CV откройте `CV_01.ipynb` (или любой другой номер).
4. Подключите Drive в первой ячейке. Выходная папка по умолчанию:
   `/content/drive/MyDrive/CV39_Sakhtay/runs`.
5. Сначала запустите `CV_29.ipynb` с `SMOKE=True`: синтетические данные не требуют
   скачивания. Этот режим проверяет пайплайн.
6. Затем используйте `SMOKE=False`. Запускайте варианты по одному или небольшими
   группами. 39 вариантов с абляциями, переносом и sweep — это много отдельных
   обучений, весь комплект не рассчитан на одну бесплатную сессию Colab.
7. При обрыве повторите ту же ячейку с теми же настройками. `last.pt` содержит
   веса, оптимизатор, scheduler, AMP scaler и RNG. Первая эпоха специально
   прерывается и продолжается для демонстрации resume.
8. После просмотра val-результатов заполните `REPORT_NOTES.md` каждого варианта,
   выберите финальную модель и выполните отдельную ячейку `FINAL_TEST=True` один раз.

Данные torchvision скачиваются автоматически; UC Merced, QuickDraw и UTKFace
нужно разместить вручную. Ни токены, ни пароли в ноутбуках не нужны.

## Структура

```text
cv39/                  # данные, модели, обучение, оценка, специальные эксперименты
notebooks/             # 39 CV-ноутбуков, стартовый и Poetry
poetry/                # исходный notebook fetch + современная адаптация fine-tuning
source/cards/          # точные карточки всех вариантов из задания
docs/                  # методические уточнения, покрытие и подготовка к защите
tests/                 # проверки данных, метрик, масок, resume и BN
runs/full/variant_NN/  # отчёты, рисунки, JSON, CSV и чекпоинты после запуска
report.py              # генерация Markdown-отчётов по реальным файлам
run.py                 # единый запуск вариантов
```

## Запуск командами

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --variant 29 --smoke --stage train --workers 0
python run.py --variant 1 --out /content/drive/MyDrive/CV39_Sakhtay/runs
python run.py --variant 2 3 --out /content/drive/MyDrive/CV39_Sakhtay/runs
python report.py --out /content/drive/MyDrive/CV39_Sakhtay/runs
```

`--stage all` включает EDA, классику, none/policy, специальный эксперимент и отчёт.
`--stage train` запускает none/policy и оценку без классики/спецэксперимента.
`--stage classical` — только классика. `--stage special` повторно загружает
базовые прогоны и запускает специальный эксперимент. Оценка каждого чекпоинта
сохраняет logits/targets, confusion matrix, per-class P/R/F1, top-3, ECE, ошибки,
Grad-CAM/attention rollout, robustness и latency.

Финальная оценка test отделена от подбора модели:

```bash
python run.py --variant 1 --stage test --out /content/drive/MyDrive/CV39_Sakhtay/runs
```

Перед test создаётся lock-файл, повторный доступ блокируется. Если сессия оборвалась
во время единственного test, сохраните причину и восстановите этот же запуск:
не удаляйте lock для подбора параметров. Выбор финальной модели должен быть
проверен по карточке: нельзя автоматически выбирать target-domain или SSL rotation
модель вместо исходного классификатора.

## Обязательные результаты

Для каждого варианта: seed=1000+номер; EDA и 3 примера на класс; лист обработки;
классический baseline; абляция none/policy; scratch и предусмотренный режим
переноса; кривые loss/accuracy/LR; resume; метрики и ошибки; robustness; карты
интерпретации; latency и учебная оценка стоимости; ответ на индивидуальный вопрос.

Основные специальные сценарии: probe→FT, few-shot с тремя seed, domain adaptation,
weighted/oversampling/focal, curriculum, sweep аугментаций/оптимизаторов/LR,
BN→GN/dropout/smoothing, KD с T², физический channel pruning + FX INT8,
rotation-prediction SSL, TTA/ensemble с ограничением бюджета capstone.
См. `docs/COVERAGE.md`: отдельные бонусы требуют дополнительного анализа и
не считаются выполненными одним наличием кода.

## Ручные датасеты

Официальные источники:
[UC Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html),
[QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset),
[UTKFace](https://susanqq.github.io/UTKFace/).
Проверяйте условия использования и возможность публикации изображений.
Сами датасеты и веса не включайте в GitHub.

Для UC Merced скопируйте классовые папки через подготовщик (оригиналы сохраняются):

```bash
python prepare_manual.py imagefolder /content/UCMerced_LandUse/Images data/ucmerced --seed 1017
python prepare_manual.py quickdraw /content/quickdraw_npy data/quickdraw --seed 1025
```

QuickDraw: десять категорий выбираются по seed, максимум 10000 рисунков на класс;
точный состав сохраняется. UTKFace: исходные jpg в `data/utkface/`, возрастные группы
0–12, 13–19, 20–35, 36–55, 56+. Подгруппы берутся только из предоставленных метаданных.
Результаты не являются основанием для реального определения возраста или решений
о людях; разметка, выборка и возрастные границы ограничивают применимость.

## Persian Poetry

Откройте `notebooks/Poetry_Colab.ipynb`. Сначала загрузите оригинал по заданной
ссылке и сохраните SHA; затем сравните ячейки с адаптацией.
Наша версия использует базовую `HooshvareLab/gpt2-fa`, а **не** готовую
`gpt2-fa-poetry` вместо обучения. Полные стихотворения разделяются до токенизации,
дубликаты удаляются; padding маскируется через attention mask, чтобы не потерять EOS.

Данные: [Chronological Persian Poetry Dataset](https://github.com/aghasemi/ChronologicalPersianPoetryDataset).
Нужно выбрать реальные названия текстовой и авторской колонок из TSV — ноутбук
сначала печатает их. Сохраняются hash данных, ревизия модели, split, loss,
perplexity, до/после генерации с одинаковыми seed/prompts и checkpoint resume.
Perplexity не доказывает художественное качество: генерации проверяются вручную.
Современная адаптация не называется точным воспроизведением старого notebook,
пока не проверены совпадение данных, формата промпта и гиперпараметров.


## Очередь всех 39 вариантов в Colab

После выполнения ячеек установки, подключения Drive и определения OUT/DATA:

```python
import subprocess, sys
FAILED = []
for number in range(1, 40):
    result = subprocess.run([sys.executable, "run.py", "--variant", str(number), "--out", OUT, "--data", DATA])
    if result.returncode != 0:
        FAILED.append(number)
print("Варианты с ошибками:", FAILED)
```

Каждый вариант запускается последовательно. Ошибка одного варианта фиксируется,
после чего очередь переходит к следующему. Для UC Merced, QuickDraw и UTKFace
нужны подготовленные данные. Poetry выполняется отдельно после выбора TSV-колонок.

## Запуск на компьютере из PowerShell

Из папки notebooks:

```powershell
Set-Location ..
python run.py --variant (1..39) --workers 0
```

Это локальное обучение, а не запуск на сервере Colab. Оно требует установленных
зависимостей и данных. CLI останавливается на первой ошибке. Очередь Colab выше
продолжает выполнение и выводит список неудачных вариантов.
