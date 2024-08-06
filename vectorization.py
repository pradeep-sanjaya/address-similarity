from sklearn.feature_extraction.text import TfidfVectorizer


def address_to_vector(address, vectorizer):
    return vectorizer.transform([address]).toarray()[0]


def add_addresses(addresses, collection):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(addresses)

    for idx, address in enumerate(addresses):
        vector = address_to_vector(address, vectorizer)
        try:
            collection.add(embeddings=[vector.tolist()], ids=[str(idx)], metadatas=[{'address': address}])
        except Exception as e:
            print(f"Error inserting address {address}: {e}")
