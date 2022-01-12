// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const express = require('express');
const handlebars = require('handlebars');
const {readFile} = require('fs').promises;
const predictorRequest = require('./predictor.js');

const app = express();
app.use(express.json());

let compiledTemplate, editorHtml;

// Load the template files and serve them with the Editor service.
const buildEditorHtml = async () => {
  try {
    compiledTemplate = handlebars.compile(
      await readFile(__dirname + '/templates/index.html', 'utf8')
    );
    editorHtml = compiledTemplate();
    return editorHtml;
  } catch (err) {
    throw Error('Error loading template: ', err);
  }
};

app.get('/', async (req, res) => {
  try {
    if (!editorHtml) editorHtml = await buildEditorHtml();
    res.status(200).send(editorHtml);
  } catch (err) {
    console.log('Error loading the Editor service: ', err);
    res.status(500).send(err);
  }
});

// The predictorRequest makes a request to the Predictor service.
// The request returns the readability excerpt, score and level
// [START cloudrun_secure_request_do]
// [START run_secure_request_do]
app.post('/predict', async (req, res) => {
  try {
    const excerpt = req.body;
    const response = await predictorRequest(excerpt);
   
    res.status(200).send(response);
  } catch (err) {
    console.log('error: markdown rendering:', err);
    res.status(500).send(err);
  }
});
// [END run_secure_request_do]
// [END cloudrun_secure_request_do]

// Exports for testing purposes.
module.exports = {
  app,
  buildEditorHtml,
};
