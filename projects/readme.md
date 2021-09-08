# Instructions

[Introduction](#Introduction).

[Objectives](#Objectives).

[Methodology](#Methodology).

[Schedule](#Schedule).

[State-of-the-Art](#State-of-the-Art).

[Semesters](#Semesters).

[References](#References)

# Introduction

The purpose of this document is to provide guidelines for the execution of projects in Deep Learning (DL), Digital Image Processing (DIP) and Computer Vision (CV) undergrad courses from the [Institute of Computing](www.ic.ufal.br) from the [Federal University of Alagoas](www.ufal.br).

In a nutshell, we propose the replication of some recent work divulged in a website called [https://paperswithcode.com/](https://paperswithcode.com/).

# Objectives

1. Stimulate discussion about recent advances in applications of DL/DIP/CV methods.
2. Stimulate students to search and implement recent solutions in DL/DIP/CV.
3. To evaluate students' abilities regarding project planning and execution.

# Methodology

1. Background search.

   - Go to [https://paperswithcode.com/](https://paperswithcode.com/) and search for some work that you find interesting/relevant for your context.

   - Your context means: possibly something you've been working on in some other project.
   - Something you're interested in working with.
   - Take into account the **feasibility** of the work you propose to replicate, particularly for DL projects which usually require a large amount of hardware computing power and manipulating large datasets.
   - Try to find something that indicates an easy reproducibility such as availability of [Google Colaboratory](https://colab.research.google.com/) notebooks and a **freely available** dataset.
   - Observe all tools and versions (Tensorflow, PyTorch, Theano, Caffe, etc), as well as all used libraries to anticipate probable difficulties in reproducing the work. One good thing is to observe how active is the repository (stars, followers, forks, wiki, commits, issues, etc).

2. Create your own repository in github where you are going to continuously update throughout the 16 weeks.

   - Upload the original repository you found on [https://paperswithcode.com/](https://paperswithcode.com/).
   - In the readme.md file, insert the reference to the original work appropriately.

3. Perform a **Background search** as follows:

   - The goal is to verify other implementations related to the original work you want to replicate.
   - A **general search** must be initially performed on Google (or other search engine) to check for general works from different sources, reference research centers on the area, related databases, relevant links, links to online videos (e.g., YouTube), possible commercial SDKs, works from amateurs and hobbyists.
   - A **specific search** must be further performed by searching scientific papers on different sources such as (Scopus, SciHub, IEEExplore, Elsevier, Springer, ACM...) and must contain recent and relevant works, specially: Articles of type Survey published in no more than the past 5 years. This specific search must be performed considering the instructions available in Section [State-of-the-Art](#State-of-the-Art).
   - Document your background search on github (**each paper must be an issue in github**).
   - Deliverable: A report containing the search results:
     - At least 3 journal papers (in the related area).
     - At least 10 conference papers (in the related area).
     - The report must contain at least one image (or possibly 2, if relevant) succinctly illustrating each paper's contribution (What is the context? Which Research Question is it tackling? What is the method? What are the results? What are its limitations?).
     - A paragraph (for each paper) with approximately 100 to 200 words long describing the paper's main contribution (similarly to an abstract).

4. Reproduce and report results from the original paper.

5. Improve the original paper with a new proposal.

6. Deliver results and final presentation.

# Schedule

The project schedule is (total of 16 weeks):

1. Week 1-4: Background search.
   - A **general search** (Week 1).
   - A **specific search** (Weeks 2-4).
2. Week 5: Introductory presentation.
   - Deliverable: One file containing a presentation (PPT or similar) containing:
     - The original paper found on [https://paperswithcode.com/](https://paperswithcode.com/).
     - The presentation of the State of The Art (SOTA) as constructed following the instructions on Section [Methodology](#Methodology).
3. Week 6-8: Replication.
   - The goal is to reproduce the original work.
   - It is important to report difficulties and overcoming them.
4. Week 9: Project presentation and (deliverable) execution.
5. Week 10-15: Project extension.
6. Week 16: Final presentation.

# State-of-the-Art

----

A good review of the literature provides the right context for your contribution:

- Complete (present main ideas, and where the current knowledge is)
- Updated (consider even unpublished works)
- Qualified (JCR journals, top conferences, high-quality books)

Conduct a systematic review: answer a well-defined research question by collecting and summarising all empirical evidence that fits pre-specified eligibility criteria. It is, thus, verifiable (open to criticism), reproducible, and replicable.

----

Scientific question: The most important element of your research project!

A good scientific question

- has not been answered yet (originality), but
- can be answered with the available resources (feasibility): time and means, and
- it matters to the community (relevance). It is also important that the progress towards the answer can be measured.

----

- The scientific question is a work-in-progress that starts with your advisor, or with colleagues, or from your own experience.
- Along with it, you should have at least n excellent references to start thinking about the problem.
- Define inclusion rules I for your review of the literature, and exclusion rules E.

----

Algorithm 1 Systematic review from cited and citing references

1. procedure Collect(P1; P2; ... ; Pn; I; E) . Find and select papers
   1. for p = 1; 2; ... ; n do
      1. Make Lp, the list of the cited and citing papers of Pp
      2. Refine Lp excluding those that comply with E, and do not comply with I
   2. end for
   3. return L = L1 ∪ L2 ∪ · · · ∪ Lp . The union of the refined lists
2. end procedure
3. while There is a change in L do
   1. Call Collect(L; I; E)
4. end while
5. return L

----

- The procedure converges fast, usually after three or four iterations (we work in a limited knowledge scenario).
- The first iteration takes the longest time.
- Use Web of Science and/or Scopus first.

----

Example

- For each paper, one can list its references and its citations:

The past -> The paper -> The future

----

Implementation I

- Web of Science allows you to create lists of references.
- You can export them in BibTEX format, and manage them using, for instance, Mendeley.
- Consider using software, e.g., QQUANT.
- Consider using VOSviewer to discover and report relationships.
- Consider using Bibliometrix for even more information.

----

The state-of-the-art **is not** a list of references, but a comprehensive analysis of the literature:

- start with a historical view, and
- group the articles by main advancements, from a conceptual viewpoint.
  A good work will make your contribution clear.

# Semesters

## 2020.2

- Under construction.

## 2020.1

- [roboticarm](https://github.com/tfvieira/roboticarm).
- [End to End Learning for Self-Driving Cars](https://paperswithcode.com/paper/end-to-end-learning-for-self-driving-cars). ([github](https://github.com/guerraepaz/tiagoprojeto)).
- [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991) ([github](https://github.com/hugotallys/bi-lstm-crf)).
- [COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest X-Ray Images](https://paperswithcode.com/paper/covid-net-a-tailored-deep-convolutional) ([github](https://github.com/CarlosW1998/COVID-Net)).
- [Practical Deep Reinforcement Learning Approach for Stock Trading](https://paperswithcode.com/paper/practical-deep-reinforcement-learning).

## 2019.2

- [Utilização de técnicas de aprendizagem profunda para estimativa de fechamento de poços verticais em rochas salinas](https://docs.google.com/presentation/d/1NopadQJb9uu8OnnLRimdAQP7iE0rrhWV/edit?usp=sharing&ouid=105639408587766163166&rtpof=true&sd=true).

- Deep Learning Aplicado ao Reconhecimento de Fala em Português do Brasil.

- Otimização de ganho de amplificadores ópticos usando inteligência artificial.

- Identificação de anomalias durante a produção de petróleo em poços _offshore_.

- Utilização de técnicas de aprendizagem prfoudna para estimativa de fechamento de poços verticias em rochas salinas;

- Previsão do comportamento de movimento de estruturas _offshore_ através da avaliação do histórico das ondas oceânicas.
- Detecção de EPIs em um ambiente fabril.

# References

## Publication sources

- [Papers with Code](https://paperswithcode.com/).

### Journals

- [IEEE Trans. Image Processing](http://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83).
- [IEE Proceedings. Vision, Image and Signal Processing](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=2200).
- [IET Computer Vision](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4159597).
- [Image and Vision Computing](https://www.sciencedirect.com/journal/image-and-vision-computing).
- [IEEE Trans. Medical Imaging](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=42).

### Conferences

- ICIP -- IEEE International Conference on Image Processing.
- MICCAI -- International Conference on Medical Image Computing and Computer Assisted Intervention.
- CAIP -- International Conference on Computer Analysis of Images and Patterns.
- ICIAP -- International Conference on Image Analysis and Processing.
- ICIAR -- International Conference on Image Analysis and Recognition.
- IWSSIP -- International Conference on Systems, Signals and Image Processing.
- SIBGRAPI -- Conference on Graphics, Patterns and Images.
- VCIP -- IEEE International Conference on Visual Communications and Image Processing.
