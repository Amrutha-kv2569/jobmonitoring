import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import os

# Scraping function
@st.cache_data(show_spinner=False)
def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{}/all/India?search={}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page, keyword.replace(" ", "%20"))
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        job_blocks = soup.find_all("div", class_="ads-details")
        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Skills": skills,
                    "Summary": summary
                })
            except:
                continue

    return pd.DataFrame(jobs_list)


# Clustering function
def cluster_jobs(df, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Skills'])

    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = model.fit_predict(X)

    # Save model and vectorizer for reuse
    with open("model.pkl", "wb") as f:
        pickle.dump((model, vectorizer), f)

    return df


# UI starts here
st.title("🔍 Karkidi Job Scraper & Skill-based Job Clustering")

keyword = st.text_input("Enter job keyword (e.g. data science, ML, AI):", "data science")
pages = st.slider("Number of pages to scrape", 1, 5, 1)
cluster_count = st.slider("Number of skill clusters", 2, 10, 5)

if st.button("Scrape & Cluster Jobs"):
    df = scrape_karkidi_jobs(keyword=keyword, pages=pages)
    if not df.empty:
        df_clustered = cluster_jobs(df, n_clusters=cluster_count)
        st.success(f"Scraped {len(df)} jobs and clustered into {cluster_count} groups.")

        st.subheader("Sample of clustered jobs")
        st.dataframe(df_clustered[['Title', 'Company', 'Skills', 'Cluster']].head(20))
    else:
        st.warning("No jobs found. Try another keyword or wait and retry.")

# Match user skills to job cluster
st.markdown("---")
st.subheader("🎯 Find Jobs Matching Your Skills")
user_skills = st.text_input("Enter your skills (comma-separated)", "python, machine learning, SQL")

if os.path.exists("model.pkl") and user_skills:
    with open("model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)

    user_vector = vectorizer.transform([user_skills])
    cluster_id = model.predict(user_vector)[0]

    df_jobs = scrape_karkidi_jobs(keyword=keyword, pages=1)
    if not df_jobs.empty:
        df_jobs['Cluster'] = model.predict(vectorizer.transform(df_jobs['Skills']))
        matched_jobs = df_jobs[df_jobs['Cluster'] == cluster_id]

        st.write(f"📌 Found {len(matched_jobs)} new jobs matching your skill cluster.")
        st.dataframe(matched_jobs[['Title', 'Company', 'Location', 'Skills']])
    else:
        st.warning("Could not load new job listings.")
