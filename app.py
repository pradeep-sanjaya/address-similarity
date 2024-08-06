import streamlit as st
import pandas as pd
from chromadb_setup import get_chromadb_client
from vectorization import add_addresses, address_to_vector
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize ChromaDB collection
collection = get_chromadb_client()

if collection is None:
    st.error("Could not connect to the ChromaDB collection. Please check the logs for more details.")
else:
    # Upload addresses
    st.title('Address Similarity Search')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        addresses_df = pd.read_csv(uploaded_file)
        if 'address' in addresses_df.columns:
            addresses = addresses_df['address'].tolist()
            add_addresses(addresses, collection)
            st.success(f"{len(addresses)} addresses added to the vector database")
        else:
            st.error("CSV file must contain an 'address' column.")

        # Search for similar addresses
        search_address = st.text_input("Enter an address to search for similar addresses")
        if search_address:
            vectorizer = TfidfVectorizer()
            vectorizer.fit(addresses_df['address'].tolist())
            search_vector = address_to_vector(search_address, vectorizer)

            try:
                results = collection.query(query_embeddings=[search_vector.tolist()], n_results=5)
                st.subheader("Similar Addresses and Scores:")

                # Debug print to inspect the structure
                # st.write(results)

                if 'metadatas' in results and 'distances' in results and results['metadatas'][0] and \
                        results['distances'][0]:
                    # Create a DataFrame to display the results in a table
                    data = []
                    for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                        data.append({"Address": metadata['address'], "Score": distance})

                    df = pd.DataFrame(data)
                    st.table(df)
                else:
                    st.error("No metadata or distances found in query results")
            except Exception as e:
                st.error(f"Error during query: {e}")
