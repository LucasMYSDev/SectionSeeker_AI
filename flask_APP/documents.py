from .models import Folder, Document
from . import db
from flask import current_app as app
from flask import Blueprint, render_template, request, jsonify,redirect, url_for,make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import pandas as pd
import sys
from flask_APP.helper_function import *
from flask_login import current_user

document_bp = Blueprint(
    "document_bp", __name__, template_folder="templates", static_folder="static"
)



@document_bp.route('/')
def index():
    return render_template('dashboard.jinja2')

@document_bp.route('/upload', methods=['POST'])
def upload_file():
    app_ = app._get_current_object()

    file = request.files['file']
    if file and file.filename.endswith('.docx'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app_.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        processed_df = prepare_document(file_path)
        parquet_path = save_to_parquet(processed_df, 'parquet_files', filename)

        processed_df_display = processed_df.copy()
        processed_df_display.drop('embeddings', axis=1, inplace=True)
        processed_df_display['text'] = processed_df_display['text'].str.replace('\n', '<br>')
        processed_df_display['section'] = processed_df_display['section'].str.replace('\n', '<br>')
        df_html_learnt = processed_df_display.to_html(escape=False, index=False)

        folder_id = request.form.get('folder_id')
        
        folder = db.session.get(Folder, folder_id)

        if not folder:
            return jsonify(success=False, message="Folder not found"), 400

        # document = Document(filename=filename, parquet_path=parquet_path, df_html=df_html_learnt, folder_id=folder_id)
        document = Document(
                filename=filename, 
                parquet_path=parquet_path, 
                df_html=df_html_learnt, 
                folder_id=folder_id,
                user_id=current_user.id  # Add this line to set the user ID
            )

        db.session.add(document)
        db.session.commit()

        return jsonify(success=True, message="File uploaded and processed successfully")
    
    else:
        return jsonify(success=False, message="Please select a valid .docx file to upload"), 400


@document_bp.route('/upload_result')
def upload_result():
    return render_template('upload_result.html')


@document_bp.route('/delete-folder/<int:folder_id>', methods=['DELETE'])
def delete_folder(folder_id):
    # folder = Folder.query.get(folder_id)  # Or use Session.get() if you've updated to SQLAlchemy 2.0
    folder = db.session.get(Folder, folder_id)

    if folder and folder.user_id == current_user.id:
        # Delete all documents in the folder
        # documents = Document.query.filter_by(folder_id=folder.id).all()
        documents = Document.query.filter_by(folder_id=folder.id, user_id=current_user.id).all()

        for doc in documents:
            file_path = doc.parquet_path
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"Warning: File {file_path} not found.")
            db.session.delete(doc)
        
        # Now delete the folder
        db.session.delete(folder)
        db.session.commit()
        return jsonify(success=True, message="Folder and its documents deleted successfully")
    else:
        return jsonify(success=False, message="Folder not found"), 404

@document_bp.route('/folders', methods=['GET'])
def get_folders():
    try:
        folders = Folder.query.filter_by(user_id=current_user.id).all()  # Filtering by current user
        return jsonify([{'id': folder.id, 'name': folder.name} for folder in folders])
    except Exception as e:
        print("Error fetching folders:", e)
        return jsonify({"error": "Failed to fetch folders"}), 500

@document_bp.route('/documents', methods=['GET'])
def documents():
    folders = Folder.query.filter_by(user_id=current_user.id).all()  # Filtering by current user
    folder_list = []
    for folder in folders:
        documents = Document.query.filter_by(folder_id=folder.id, user_id=current_user.id).all()  # Filtering by current user and folder
        document_list = [{'id': document.id, 'filename': document.filename} for document in documents]
        folder_list.append({'folder_id': folder.id, 'folder_name': folder.name, 'documents': document_list})

    all_documents = Document.query.filter_by(user_id=current_user.id).all()  # Filtering by current user
    all_documents_list = [{"id": doc.id, "filename": doc.filename} for doc in all_documents]

    return jsonify({"folders": folder_list, "all_documents": all_documents_list})


@document_bp.route('/view-learnt-documents/<int:document_id>')
def view_learnt_doc(document_id):
    document = Document.query.get(document_id)
    if document and document.df_html:
        # Render a template and pass the table to it
        return render_template('view_learnt_doc.html', table=document.df_html)
    else:
        return "File not found", 404

@document_bp.route('/create-folder', methods=['POST'])
def create_folder():
    name = request.json['name']

    # Check if a folder with this name already exists for the user
    existing_folder = Folder.query.filter_by(name=name, user_id=current_user.id).first()  # Filtering by current user
    if existing_folder:
        return jsonify(success=False, message="A folder with this name already exists"), 400

    # Create the new folder for the user
    folder = Folder(name=name, user_id=current_user.id)  # Set the user ID
    db.session.add(folder)
    db.session.commit()

    return jsonify(folder_id=folder.id, folder_name=folder.name, documents=[])

@document_bp.route('/ask', methods=['POST'])
def ask_question():
    document_ids = request.json.get('document_ids')
    question = request.json.get('question')

    if not document_ids:
        return jsonify(success=False, message="No document IDs provided"), 400

    dfs = []  # This will store dataframes for each document

    for doc_id in document_ids:
        document = Document.query.get(doc_id)
        if not document:
            return jsonify(success=False, message=f"Document with ID {doc_id} not found"), 404

        # Load the processed document data
        parquet_path = document.parquet_path
        parquet_directory = os.path.dirname(parquet_path)
        parquet_filename = os.path.basename(parquet_path)

        processed_df = load_from_parquet(parquet_directory, parquet_filename)
        dfs.append(processed_df)

        response_df = ask_question_mutiple_doc_and_return_df_response_drop_duplicate(question, dfs)
        user_question = question  

    answer = response_df['Data'].iloc[0]
    response_df['Data'] = response_df['Data'].str.replace('\n', '<br>')
    response_df['Section Name'] = response_df['Section Name'].str.replace('\n', '<br>')

    df_html = response_df.to_html(index=False)
    return jsonify(success=True, user_question=user_question, answer=answer, df_html=df_html)

@document_bp.route('/delete-document/<int:document_id>', methods=['DELETE'])
def delete_document(document_id):
    document = Document.query.get(document_id)  # Or use Session.get() if you've updated to SQLAlchemy 2.0
    if document and document.user_id == current_user.id:
        file_path = document.parquet_path
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"Warning: File {file_path} not found.")

        db.session.delete(document)
        db.session.commit()
        return jsonify(success=True, message="Document deleted successfully")
    else:
        return jsonify(success=False, message="Document not found"), 404


