index_mapping='
{
  "mappings": {
    "properties": {
      "patient_lastname": {
        "type": "keyword"
      },
      "patient_firstname": {
        "type": "keyword"
      },
      "text": {
        "type": "text",
        "analyzer": "standard"
      },
      "date": {
        "type": "date"
      },
      "patient_left": {
        "type": "boolean"
      },
      "emotion": {
        "type": "keyword"
      },
      "confidence": {
        "type": "float"
      }
    } 
  }
}'

# Utilisez la variable index_mapping dans la commande cURL
curl -X PUT 'http://127.17.0.1:9200/notes' -H 'Content-Type: application/json' -d "$index_mapping"
