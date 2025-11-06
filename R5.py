import requests
from bs4 import BeautifulSoup
import numpy as np

# Step 1: Define some real pages (small network for demonstration)
pages = {
    "https://www.python.org": [],
    "https://docs.python.org/3/": [],
    "https://pypi.org/": [],
    "https://www.djangoproject.com/": []
}

# Step 2: Extract all hyperlinks from each page (keeps only links inside our small set)
for page in pages.keys():
    try:
        print(f"Fetching links from: {page}")
        response = requests.get(page, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        links = [a['href'] for a in soup.find_all('a', href=True)]
        # Keep only internal links among our pages
        links = [link for link in links if link.startswith("https://www.python.org")
                 or link.startswith("https://docs.python.org")
                 or link.startswith("https://pypi.org")
                 or link.startswith("https://www.djangoproject.com")]
        # remove duplicates while preserving order (optional)
        seen = set()
        pages[page] = [x for x in links if not (x in seen or seen.add(x))]
    except Exception as e:
        print(f"Error fetching {page}: {e}")

# Step 3: Build the column-stochastic link matrix (M) where M[j,i] = prob of i -> j
n = len(pages)
page_list = list(pages.keys())
link_matrix = np.zeros((n, n), dtype=float)

for i, page in enumerate(page_list):
    # only consider out-links that are in our page_list
    out_links = [l for l in pages[page] if l in page_list]
    if len(out_links) > 0:
        prob = 1.0 / len(out_links)
        for link in out_links:
            j = page_list.index(link)
            link_matrix[j, i] = prob
    else:
        # Dangling node: distribute uniformly to all pages
        link_matrix[:, i] = 1.0 / n

# Step 4: Power iteration for PageRank
damping_factor = 0.85
num_iterations = 100
tol = 1e-8

rank = np.ones(n) / n
for it in range(num_iterations):
    new_rank = (1 - damping_factor) / n + damping_factor * (link_matrix.dot(rank))
    if np.linalg.norm(new_rank - rank, 1) < tol:
        rank = new_rank
        print(f"Converged after {it+1} iterations.")
        break
    rank = new_rank

# Step 5: Display PageRank values
print("\nFinal PageRank Scores:")
for i, page in enumerate(page_list):
    print(f"{page}: {rank[i]:.6f}")
