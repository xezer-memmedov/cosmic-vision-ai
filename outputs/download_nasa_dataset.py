import os
import csv
import requests

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
SOURCES_PATH = os.path.join(PROJECT_DIR, "dataset_sources.csv")

SEARCH_QUERIES = {
    "galaxy": "galaxy Hubble",
    "nebula": "nebula Hubble",
    "star_cluster": "star cluster Hubble",
}

IMAGES_PER_CLASS = 20


def download_images(class_name, query):
    class_dir = os.path.join(DATASET_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    response = requests.get(
        "https://images-api.nasa.gov/search",
        params={
            "q": query,
            "media_type": "image",
            "page_size": IMAGES_PER_CLASS,
        },
        timeout=30,
    )
    response.raise_for_status()

    items = response.json()["collection"]["items"]
    downloaded = 0
    source_rows = []

    for item in items:
        if downloaded >= IMAGES_PER_CLASS:
            break

        data = item["data"][0]
        links = item.get("links", [])

        image_url = next(
            (
                link["href"]
                for link in links
                if link.get("render") == "image"
            ),
            None,
        )

        if image_url is None:
            continue

        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()

            file_name = f"{class_name}_{downloaded:03d}.jpg"
            file_path = os.path.join(class_dir, file_name)

            with open(file_path, "wb") as image_file:
                image_file.write(image_response.content)

            source_rows.append(
                [
                    class_name,
                    file_name,
                    data.get("nasa_id", ""),
                    data.get("title", ""),
                    image_url,
                ]
            )

            downloaded += 1
            print(f"Yuklendi: {file_name}")

        except requests.RequestException:
            print("Bir sekil yuklenmedi, digerine kecilir.")

    return source_rows


all_sources = []

for class_name, query in SEARCH_QUERIES.items():
    print(f"\n{class_name} sinfi yuklenir...")
    all_sources.extend(download_images(class_name, query))

with open(SOURCES_PATH, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(
        ["class_name", "file_name", "nasa_id", "title", "source_url"]
    )

    writer.writerows(all_sources)

print("\nDataset hazirdir.")
print(f"Toplam sekil sayi: {len(all_sources)}")
print(f"Menbeler saxlanildi: {SOURCES_PATH}")