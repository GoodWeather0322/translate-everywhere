# python packages
import logging
import asyncio
import aiofiles
from pathlib import Path
from datetime import datetime
import os
import base64

# FastAPI dependencies
from fastapi import APIRouter, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse

# project package
from core.end2end import AzureEnd2End


# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


router = APIRouter()
model = AzureEnd2End()

@router.get("/get-support-languages")
async def get_support_languages():
    """
    Retrieve the list of supported languages for translation.

    Returns:
        A list of language codes that the translation model supports.
    """
    return model.get_support_languages()

@router.get("/get-custom-models")
async def get_custom_models():
    """
    Retrieve the list of custom models.

    Returns:
        A list of custom models.
    """
    return model.get_custom_models()


@router.post("/speech-translate")
async def speech_translate(file: UploadFile = File(..., description="audio file to be translated"), 
                           source_lang: str = Form(..., description="source language code"), 
                           target_lang: str = Form(..., description="target language code")):
    """
    This function handles the speech translation endpoint.
    
    Parameters form input:
    - file: Uploaded audio file
    - source_lang: Source language for translation
    - target_lang: Target language for translation
    
    Returns:
    JSONResponse with source text [source_text], target text[target_text], and base64 audio file [file]
    """
    logger.info("speech-translate endpoint connected!")
    logger.info(f"source_lang: {source_lang}")
    logger.info(f"target_lang: {target_lang}")
    logger.info(f"*****************")
    
    save_path = Path("uploaded_audio")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    date = timestamp[:8]
    save_path = save_path / date
    save_path.mkdir(parents=True, exist_ok=True)
    file_extension = Path(file.filename).suffix
    file_name = f"{timestamp}{file_extension}"

    async with aiofiles.open(
        save_path / file_name, "wb"
    ) as out_file:
        # Read the file in chunks and write to the output file
        while content := await file.read(1024):
            await out_file.write(content)

    source_text, target_text, output_file = await asyncio.to_thread(model.sts_end2end_pipeline, source_lang, target_lang, str(save_path / file_name))
    logger.info(source_text)
    logger.info(target_text)

    if output_file is not None:
        with open(output_file, "rb") as file:
            encoded_file = base64.b64encode(file.read()).decode("utf-8")
    else:
        encoded_file = None
    # 返回 WAV 檔案和文本
    return JSONResponse(content={
        "source_text": source_text,
        "target_text": target_text,
        "file": encoded_file
    })

@router.post("/speech-translate-custom")
async def speech_translate_custom(file: UploadFile = File(..., description="audio file to be translated"), 
                                  source_lang: str = Form(..., description="source language code"), 
                                  target_lang: str = Form(..., description="target language code"), 
                                  name: str = Form(..., description="custom model name: ['auto', 'evonne', 'laura']")):
    """
    This function handles the speech translation endpoint with a custom name parameter.
    
    Parameters form input:
    - file: Uploaded audio file
    - source_lang: Source language for translation
    - target_lang: Target language for translation
    - name: Custom name for the translation session
    
    Returns:
    JSONResponse with source text [source_text], target text[target_text], and base64 audio file [file]
    """
    logger.info("speech-translate-custom endpoint connected!")
    logger.info(f"source_lang: {source_lang}")
    logger.info(f"target_lang: {target_lang}")
    logger.info(f"name: {name}")
    logger.info(f"*****************")
    
    save_path = Path("uploaded_audio")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    date = timestamp[:8]
    save_path = save_path / date
    save_path.mkdir(parents=True, exist_ok=True)
    file_extension = Path(file.filename).suffix
    file_name = f"{timestamp}{file_extension}"

    async with aiofiles.open(
        save_path / file_name, "wb"
    ) as out_file:
        # Read the file in chunks and write to the output file
        while content := await file.read(1024):
            await out_file.write(content)

    source_text, target_text, output_file = await asyncio.to_thread(model.sts_end2end_pipeline, source_lang, target_lang, str(save_path / file_name), name)
    logger.info(source_text)
    logger.info(target_text)

    if output_file is not None:
        with open(output_file, "rb") as file:
            encoded_file = base64.b64encode(file.read()).decode("utf-8")
    else:
        encoded_file = None
    # 返回 WAV 檔案和文本
    return JSONResponse(content={
        "source_text": source_text,
        "target_text": target_text,
        "file": encoded_file
    })

@router.post("/text-translate-custom")
async def text_translate_custom(source_text: str = Form(..., description="source text to be translate"), 
                                  source_lang: str = Form(..., description="source language code"), 
                                  target_lang: str = Form(..., description="target language code"), 
                                  name: str = Form(..., description="custom model name: ['auto', 'evonne', 'laura']")):
    """
    This function handles the speech translation endpoint with a custom name parameter.
    
    Parameters form input:
    - file: Uploaded audio file
    - source_lang: Source language for translation
    - target_lang: Target language for translation
    - name: Custom name for the translation session
    
    Returns:
    JSONResponse with source text [source_text], target text[target_text], and base64 audio file [file]
    """
    logger.info("text-translate-custom endpoint connected!")
    logger.info(f"source_lang: {source_lang}")
    logger.info(f"target_lang: {target_lang}")
    logger.info(f"name: {name}")
    logger.info(f"*****************")
    
    save_path = Path("uploaded_audio")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    date = timestamp[:8]
    save_path = save_path / date
    save_path.mkdir(parents=True, exist_ok=True)
    file_extension = '.wav'
    file_name = f"{timestamp}{file_extension}"

    source_text, target_text, output_file = await asyncio.to_thread(model.tts_end2end_pipeline, source_lang, target_lang, source_text, str(save_path / file_name), name)
    logger.info(source_text)
    logger.info(target_text)

    if output_file is not None:
        with open(output_file, "rb") as file:
            encoded_file = base64.b64encode(file.read()).decode("utf-8")
    else:
        encoded_file = None
    # 返回 WAV 檔案和文本
    return JSONResponse(content={
        "source_text": source_text,
        "target_text": target_text,
        "file": encoded_file
    })