# python packages
import logging
import asyncio
import aiofiles
from pathlib import Path
from datetime import datetime
import os

# FastAPI dependencies
from fastapi import APIRouter, WebSocketDisconnect, File, UploadFile
from fastapi.responses import FileResponse

# project package
from core.end2end import AzureEnd2End


# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


router = APIRouter()
model = AzureEnd2End()


@router.post("/speech-translate")
async def speech_translate(file: UploadFile = File(...)):
    logger.info("speech-translate endpoint connected!")
    
    save_path = Path("uploaded_audio")
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_extension = Path(file.filename).suffix
    file_name = f"{timestamp}{file_extension}"

    async with aiofiles.open(
        save_path / file_name, "wb"
    ) as out_file:
        # Read the file in chunks and write to the output file
        while content := await file.read(1024):
            await out_file.write(content)

    output_file = model.end2end_flow('zh', 'en', str(save_path / file_name))
    print(output_file)
    if not os.path.abspath(output_file):
        return output_file
    try:
        output_file = Path(output_file)
        return FileResponse(path=output_file, filename=output_file.name, media_type="audio/wav")
    except:
        return output_file
