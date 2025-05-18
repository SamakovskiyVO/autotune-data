# AutoVisionTune — Data-Generation Pipeline

Полный набор скриптов для **сборки и расширения** набора изображений,
используемого в ВКР для классификации качества кадров на GoA-4 поезде.

## Возможности
* Классические аугментации (Albumentations) **+** физически-мотивированные
  эффекты (туман, ливень, …).
* **Диффузионная генерация**:  
  * *Stable Diffusion XL* (через `diffusers`) — офлайн.  
  * *Yandex GPT Vision/Diffusion API* — онлайн-расширение редких классов.
* Полная воспроизводимость: `seed` в config.
* CLI: `python -m src.cli full --config config.yaml`.
* Тесты pytest.

## Быстрый старт
```bash
git clone https://example.com/autotune-data.git
cd autotune-data
./setup.sh
python -m src.cli full --config config.yaml
python -m src.cli sample --n 12
