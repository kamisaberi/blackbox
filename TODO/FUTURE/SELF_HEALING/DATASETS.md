For **Self-Healing Infrastructure (IaC Security)**, you cannot use standard log datasets. You need **Code Datasets**—specifically, datasets that contain **"Insecure Infrastructure Code"** paired with **"Secure Fixed Code."**

Unlike network traffic (PCAP), there isn't one single massive CSV file for this. Instead, the industry relies on **"Vulnerable-by-Design" Repositories** (Goats) and **Static Analysis Rule Banks**.

Here are the resources you need to train and test "The Architect."

---

### **1. The "Goat" Projects (Vulnerable Environments)**
*Best for: End-to-End Testing of your Self-Healing Logic.*

These are open-source repositories intentionally filled with security misconfigurations. You point your Blackbox Architect at these repos to see if it can find and fix the bugs.

| Dataset / Repo | Technology | What it contains | Download Link |
| :--- | :--- | :--- | :--- |
| **TerraGoat** | **Terraform** | AWS/Azure/GCP Terraform files with open ports, unencrypted DBs, and public buckets. | [GitHub - bridgecrewio/terragoat](https://github.com/bridgecrewio/terragoat) |
| **Kubernetes Goat** | **Kubernetes** | Vulnerable YAML manifests (Privileged containers, missing limits). | [GitHub - madhuakula/kubernetes-goat](https://github.com/madhuakula/kubernetes-goat) |
| **CfnGoat** | **CloudFormation** | AWS CloudFormation templates with built-in security flaws. | [GitHub - bridgecrewio/cfngoat](https://github.com/bridgecrewio/cfngoat) |
| **Docker-Vulnerable** | **Dockerfile** | Dockerfiles running as root, exposing wrong ports, using old base images. | [GitHub - vuls/vuls-test-server](https://github.com/vuls/vuls-test-server) |

**How to use:**
1.  Fork `TerraGoat`.
2.  Run Blackbox Architect against it.
3.  **Success Metric:** Did Blackbox generate 50+ Pull Requests to fix the issues?

---

### **2. The "Rule Banks" (Training Data for AI)**
*Best for: Training your LLM/Classifier to recognize "Bad" vs "Good" code.*

Open-source security scanners (like Checkov or KICS) contain massive folders of test data to verify their rules. These folders usually contain pairs: `positive.tf` (Secure) and `negative.tf` (Insecure). **This is your labeled dataset.**

| Source | Description | Where the Data is Hidden | Link |
| :--- | :--- | :--- | :--- |
| **Checkov (Palo Alto)** | The industry standard IaC scanner. | Look inside `/tests/terraform` or `/checkov/terraform/checks/resource/aws`. | [GitHub](https://github.com/bridgecrewio/checkov) |
| **KICS (Checkmarx)** | Scans Terraform, K8s, Docker, Ansible. | Look inside `/assets/queries`. It contains thousands of samples. | [GitHub](https://github.com/Checkmarx/kics) |
| **Trivy (Aqua)** | Container and filesystem scanner. | Look at their "Policies" or Rego library tests. | [GitHub](https://github.com/aquasecurity/trivy) |

**How to create a Dataset from this:**
1.  Clone the **Checkov** repo.
2.  Write a Python script to crawl the `tests` directory.
3.  Extract `pass.tf` (Label: Safe) and `fail.tf` (Label: Vulnerable).
4.  **Result:** You now have ~2,000 labeled pairs of infrastructure code to fine-tune your model.

---

### **3. Academic & Large Scale Datasets**
*Best for: Generalization.*

| Dataset | Description | Use Case |
| :--- | :--- | :--- |
| **Project KB (SAP)** | A knowledge base of vulnerabilities mapped to code commits. | Understanding how code fixes relate to vulnerabilities. | [GitHub](https://github.com/SAP/project-kb) |
| **Google BigQuery GitHub** | The public dataset of all GitHub code. | Use SQL to scrape all files ending in `.tf` that contain `ingress { cidr_blocks = ["0.0.0.0/0"] }`. | [Google Cloud](https://cloud.google.com/blog/topics/public-datasets/github-on-bigquery-analyze-all-the-open-source-code) |
| **IAC-Security-Dataset** | A curated academic collection of Puppet/Chef/Ansible scripts with smells. | Researching configuration drift. | [GitHub (Rahman et al)](https://github.com/akondrahman/iac-security-dataset) |

---

### **4. How to Generate "Synthetic" Fix Data**

Since "Fixing" code is hard to find in datasets, you should generate your own using **GPT-4** or **CodeLlama** to create the training data for your smaller, local model.

**The "Synthetic Generator" Script:**

```python
# blackbox-sim/scripts/generate_iac_fixes.py
import openai
import os

# Prompt Engineering
PROMPT = """
I will give you a piece of INSECURE Terraform code.
You must rewrite it to be SECURE based on CIS Benchmarks.
Output ONLY the fixed code.

Input:
resource "aws_security_group" "bad" {
  ingress {
    from_port = 22
    to_port = 22
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
"""

def generate_fix():
    # Call OpenAI API (or local Ollama)
    # Save Input (Bad) and Output (Good) to a JSONL file
    pass
    
# Result: training_data.jsonl
# {"prompt": "resource aws_s3_bucket...", "completion": "resource aws_s3_bucket... acl='private'..."}
```

### **Strategic Recommendation**

1.  **Don't train a model from scratch.** Infrastructure code syntax (HCL, YAML) is strict.
2.  **Use Rule-Based Patching First.** Use the library **`hclwrite`** (Go) or **`python-hcl2`**.
    *   Logic: `Find resource "aws_s3_bucket" -> If "acl" is missing, Inject "acl = private"`.
    *   This is 100% accurate and safer than an LLM.
3.  **Use LLMs for the "Description".** Use the AI to write the **Pull Request Comment** explaining *why* the fix is needed (e.g., "Closing this port prevents SSH brute forcing"), but use deterministic code to make the fix.

**Start with `TerraGoat`.** If Blackbox can fix TerraGoat, you have a working product.


# RELATED DATASETS

