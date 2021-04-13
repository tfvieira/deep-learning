<!-- sota.md -->
# Projects

----

## Publication sources

### Journals
- [Papers with Code](https://paperswithcode.com/).
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
        1. Refine Lp excluding those that comply with E, and do not comply with I
    1. end for
    1. return L = L1 ∪ L2 ∪ · · · ∪ Lp . The union of the refined lists
1. end procedure
1. while There is a change in L do
    1. Call Collect(L; I; E)
1. end while
1. return L

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