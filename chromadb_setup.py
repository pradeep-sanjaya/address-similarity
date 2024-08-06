import chromadb


def get_chromadb_client():
    try:
        client = chromadb.Client()
        collection_name = 'address_vectors'

        # Attempt to create the collection
        try:
            print("Creating collection...")
            client.create_collection(collection_name)
        except Exception as create_e:
            print(f"Error creating collection: {create_e}")

        # Attempt to get the collection
        try:
            print("Getting collection...")
            collection = client.get_collection(collection_name)
            print("Collection retrieved successfully")
        except Exception as get_e:
            print(f"Error getting collection: {get_e}")
            return None

        return collection

    except Exception as e:
        print(f"Error initializing ChromaDB client: {e}")
        return None
