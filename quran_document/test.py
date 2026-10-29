from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

query = "query: Bagaimana sikap terhadap orang miskin?"

documents = [
    {
        "label": "R1",
        "reference": "QS. Al-Isra: 26",
        "text": "Dan berikanlah kepada keluarga-keluarga yang dekat akan haknya, kepada orang miskin dan orang yang dalam perjalanan dan janganlah kamu menghambur-hamburkan (hartamu) secara boros."
    },
    {
        "label": "R2",
        "reference": "QS. An-Nisa: 36",
        "text": "Sembahlah Allah dan janganlah kamu mempersekutukan-Nya dengan sesuatupun. Dan berbuat baiklah kepada dua orang ibu-bapa, karib-kerabat, anak-anak yatim, orang-orang miskin, tetangga yang dekat dan tetangga yang jauh, dan teman sejawat, ibnu sabil dan hamba sahayamu. Sesungguhnya Allah tidak menyukai orang-orang yang sombong dan membangga-banggakan diri,"
    },
    {
        "label": "R3",
        "reference": "QS. Al-Ma'un: 3",
        "text": "dan tidak menganjurkan memberi makan orang miskin."
    },
    {
        "label": "N1",
        "reference": "QS. Al-Baqarah: 275",
        "text": "Orang-orang yang makan (mengambil) riba tidak dapat berdiri melainkan seperti berdirinya orang yang kemasukan syaitan lantaran (tekanan) penyakit gila. Keadaan mereka yang demikian itu, adalah disebabkan mereka berkata (berpendapat), sesungguhnya jual beli itu sama dengan riba, padahal Allah telah menghalalkan jual beli dan mengharamkan riba."
    },
    {
        "label": "N2",
        "reference": "QS. Al-Anbiya: 30",
        "text": "Dan apakah orang-orang yang kafir tidak mengetahui bahwasanya langit dan bumi itu keduanya dahulu adalah suatu yang padu, kemudian Kami pisahkan antara keduanya. Dan dari air Kami jadikan segala sesuatu yang hidup. Maka mengapakah mereka tiada juga beriman?"
    }
]

document_texts = [
    f"passage: {document['reference']}. {document['text']}"
    for document in documents
]

document_embeddings = model.encode(
    document_texts,
    normalize_embeddings=True
)

query_embedding = model.encode(
    query,
    normalize_embeddings=True
)

scores = document_embeddings @ query_embedding

results = sorted(
    zip(documents, scores),
    key=lambda item: item[1],
    reverse=True
)

for document, score in results:
    print(
        f"{document['label']} | "
        f"{score:.4f} | "
        f"{document['reference']}"
    )