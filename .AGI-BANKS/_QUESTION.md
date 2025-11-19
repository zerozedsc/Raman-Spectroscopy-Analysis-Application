# *2025-11-18*

1. We will focus on enhancing `analysis_page.py` based on the following clear requirements and directives:

    - Upon opening or navigating to the analysis page, the user will immediately see a startup view that presents all available analysis methods, accompanied by concise descriptions for each. We will adopt a card block layout (refer to Image 1) and categorize the analysis methods accordingly. Each card will feature a button to initiate the respective analysis method.

    - Clicking on an analysis method card will transition the user to that specific method's view (refer to Image 2). This view will present a comprehensive input form tailored to the selected analysis method, complete with all necessary input fields. A back button will be prominently displayed to allow users to return to the startup view effortlessly.

    - After the user completes the input form and submits it, the analysis results will be displayed directly in the same view, beneath the input form.

    - We will implement a top bar with a plus button enabling users to quickly start a new analysis method from any location on the analysis page. Pressing the plus button will return the user to the startup view.

    - A sidebar will be included to showcase the history of analysis methods completed by the user during the current session. Users will be able to click on any history item to revisit both the input and results of previous analyses.

    - We will also include a feature that allows users to save the analysis results as either a data file or a graph image file. An export button will be placed below the analysis results for this purpose.

    - Different views will be developed for various analysis methods, as they will require distinct input fields and result display formats. For example, PCA will necessitate a parameter for `n_components`, while clustering will focus on `n_clusters`. We will create a flexible layout that accommodates these differences seamlessly.

    - Rigorous error handling and validation will be integrated for all input fields to ensure that user inputs are valid prior to the submission of any analysis request.

    - We will address the performance of the analysis methods, recognizing that some may require longer processing times. A loading indicator will be implemented to keep users informed that the analysis is in progress.

    - Finally, we will ensure that the analysis page is fully responsive, guaranteeing optimal functionality across various screen sizes and devices.

2. Also dont forget string localization as this is important for our global user base. All text elements on the analysis page should be adaptable to multiple languages to enhance accessibility and user experience worldwide. Right now focus on Japanese and English localization.
