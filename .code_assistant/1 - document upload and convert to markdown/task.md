# Overview

We already created backend (python fast api) and frontend (react) projects, please based on the existing solution design patterns and existing functions to implement the document upload and convert to markdown features.

# Requirements

1. Uploaded file saved into server side folder called input
2. Convert the uploaded file to markdown file and save it into server side folder called output
3. Show uploaded files info on home page the documents list grid. If its markdown file is generated, show a view button to view the markdown file content.
4. If the markdown file is not generated for the uploaded file, show a convert button to convert the file to markdown file in each file record row in the grid.

# Development

1. in the home page, we allow upload doc/excel/pdf/image files. uploaded files saved into server side folder called input.
2. If the upload file is pdf/image, call python backend service ocr_service/parser.py methods to convert the file to markdown file and save it into server side folder called output. Note: We already have sample markdown files generaed in output folder, the sub folder name based on the upload file name. You can try to link the ###\_nohf.md file as the markdown file content.
3. Note: The markdown file convert process is time cosuming, we need show a progress bar to indicate the convert progress in front end. (Note: You may need to design a independent process to handle the convert process in backend side, further, design a websocket to push progress update to front end.)
4. As for uploaded doc/excel files, we can show a convert button in each file record row in the grid. When click the convert button, call python backend service ocr_service/parser.py methods to convert the file to markdown file and save it into server side folder called output. (On hold this process at this moment)
