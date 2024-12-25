#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:26:47 2024

@author: mariyajeeranwala
"""


'''
- app.py
- images/ (folder containing product images)
- saved_updated.csv (your dataset)
'''

import streamlit as st
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = pd.read_csv("/Users/mariyajeeranwala/Desktop/ikea/data.csv")

# Pagination settings
images_per_page = 24  # Display 24 images per page
total_pages = len(data) // images_per_page + (len(data) % images_per_page > 0)

# Page selection
page = st.number_input("Page", min_value=1, max_value=total_pages, step=1, label_visibility="visible")
start_idx = (page - 1) * images_per_page
end_idx = min(start_idx + images_per_page, len(data))

# Paginated data
paginated_data = data.iloc[start_idx:end_idx]

# Display product gallery
st.title("IKEA Product Recommender")
st.subheader(f"Showing Page {page} of {total_pages}")
cols = st.columns(4)  # Display images in 4 columns (6 rows for 24 images)

# Persistent state for selected image
if "selected_image_name" not in st.session_state:
    st.session_state.selected_image_name = None

for i, row in paginated_data.iterrows():
    image_path = os.path.join("all_images", row['image_name'])
    with cols[i % 4]:
        st.image(image_path, caption=row['image_name'], use_column_width=True)
        if st.button(f"Select {row['image_name']}", key=f"btn_{row['image_name']}"):
            st.session_state.selected_image_name = row['image_name']



# Main Section: Show selected image and recommendations
if st.session_state.selected_image_name:
    selected_image_name = st.session_state.selected_image_name

    # Display selected image
    st.subheader(f"Viewing: {selected_image_name}")
    selected_image_path = os.path.join("all_images", selected_image_name)
    st.image(selected_image_path, caption="Selected Product", use_column_width=True)

    # Helper: Get recommendations within the same cluster
    def get_recommendations_within_cluster(selected_image_name, data, top_n=4):
        # Get selected image's cluster and features
        selected_row = data[data['image_name'] == selected_image_name]
        selected_features = selected_row.iloc[:, 0:3].values
        selected_cluster = selected_row["cluster"].iloc[0]

        # Filter data to the same cluster
        cluster_data = data[data["cluster"] == selected_cluster]

        # Compute cosine similarities within the cluster
        features = cluster_data.iloc[:, 0:3].values
        similarities = cosine_similarity(selected_features, features).flatten()

        # Rank by similarity
        cluster_data = cluster_data.copy()  # Avoid modifying original data
        cluster_data["similarity"] = similarities
        recommendations = cluster_data.sort_values("similarity", ascending=False).head(top_n)
        return recommendations

    # Generate recommendations
    st.subheader("Recommended Products")
    recommendations = get_recommendations_within_cluster(selected_image_name, data)

    # Number of columns to display the recommendations
    images_per_row = 4
    cols = st.columns(images_per_row)

    # Loop through the recommendations and display them
    for i, row in recommendations.iterrows():
        # Get the image path
        recommended_image_path = os.path.join("all_images", row['image_name'])
        
        # Calculate which column to use for this image
        col_index = i % images_per_row
        with cols[col_index]:
            st.image(recommended_image_path, caption=row['image_name'], use_column_width=True)


