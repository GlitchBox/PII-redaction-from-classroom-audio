import sys
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from gtts import gTTS 
import assemblyai as aai

# replace with your API token
aai.settings.api_key = f"478a4d18206446ad9ebaa3deee150da8"
transcriptFile = "transcript.txt"
redactedTranscript = "redacted_transcript.txt"
outputAudio = "PII_replaced"


def generateTranscript(audioFile):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audioFile)

    with open(transcriptFile, "w") as f:
        f.write(transcript.text)

def textToSpeech(transcript, outputFileName):

    language = 'en'
    myobj = gTTS(text=transcript, lang=language, slow=False) 
    myobj.save(outputFileName + ".mp3") 

def PII_identifier(fileName):

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    ner = pipeline("ner", model=model, tokenizer=tokenizer)
    transcription = ""

    with open(fileName, "r") as file:
        transcription = file.read()

    output = ner(transcription)

    # for entity in output:
    #     print(entity['entity'] + " " + entity['word'])
    uniqueEntities = {}
    for entity in output:
        if entity['word'] not in uniqueEntities:
            uniqueEntities[entity['word']] = entity['entity']

    for entity in uniqueEntities:
        print(entity + ": " + uniqueEntities[entity])

    entityTypes = {}
    for entity in uniqueEntities:
        
        if uniqueEntities[entity] not in entityTypes:
            entityTypes[uniqueEntities[entity]] = [entity]
        else:
            entityTypes[uniqueEntities[entity]].append(entity)
    
    print("\n")
    for eType in entityTypes:
        print(eType + ": " + str(entityTypes[eType]))
    print()

    return entityTypes, transcription


def replacePII(entityTypes, transcription):

    for eType in entityTypes:
        template = ""

        if "PER" in eType:
            template = "person"
        elif "LOC" in eType:
            template = "place"
        else:
            template = "organization" 

        idx = 1
        for entity in entityTypes[eType]:
            if "I-" in eType:
                # print("removing " + eType)
                if "#" in entity:
                    transcription = transcription.replace(" "+entity[-1],"")
                else:
                    transcription = transcription.replace(entity, "")
            else:
                # print("replacing "+eType)
                transcription = transcription.replace(entity, template+ str(idx) + " ")
                idx += 1
    return transcription



if __name__ == "__main__":
    audioFile = sys.argv[1]
    generateTranscript(audioFile)

    entityTypes, transcriptionString = PII_identifier(transcriptFile)
    redacted_transcript = replacePII(entityTypes, transcriptionString)

    with open(redactedTranscript, "w") as file:
        file.write(redacted_transcript)

    with open(redactedTranscript, "r") as file:
        redacted_transcript = file.read()
    
    textToSpeech(redacted_transcript, outputAudio)