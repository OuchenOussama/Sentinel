import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component, OnDestroy, OnInit } from '@angular/core';


@Component({
  templateUrl: './ecommerce.dashboard.component.html',
  styleUrls: ['./ecommerce.dashboard.component.scss']
})


export class EcommerceDashboardComponent {

  url = 'http://localhost:5000';
  video_path = '';

  analyserDisabled: boolean = true
  resultsDisabled: boolean = true

  analyseLoading: boolean = false;
  resulLoading: boolean = false;

  results: any[] = []

  data: any;
  options: any;

  images : any;

  selectedFile: File | undefined;

  constructor(private http: HttpClient) { }

  onUpload(event: any) {
    this.selectedFile = event.target.files[0] as File;
    this.uploadVideo()
  }


  async uploadVideo() {
    if (!this.selectedFile) {
      console.error('No file selected');
      return;
    }

    const formData = new FormData();
    formData.append("video", this.selectedFile);

    this.http.post(this.url + "/upload", formData).subscribe(
      (res: any) => {
        this.video_path = res.video_path;
        this.analyserDisabled = false;
        this.resultsDisabled = true;
        this.images = []
      },
      (error) => {
        console.error('Error !!!', error);
      }
    );
  }

  extractEmotions() {
    this.analyseLoading = true;

    if (this.video_path.length <= 0) {
      console.error('Upload a video first');
      return;
    }

    const headers = new HttpHeaders();
    headers.set('Content-Type', 'application/json');

    this.http.post(this.url + "/extract_emotions", { video_path: this.video_path }, { headers: headers }).subscribe(
      (res: any) => {
        if (res.message == "Emotions extracted successfully") {
          this.resultsDisabled = false;
          this.getResults()
        }
        this.analyseLoading = false;
      },
      (error) => {
        console.error('Error !!!', error);
        this.analyseLoading = false;
      }
    );
  }

  getResults() {
    const documentStyle = getComputedStyle(document.documentElement);
    const textColor = documentStyle.getPropertyValue('--text-color');
    const textColorSecondary = documentStyle.getPropertyValue('--text-color-secondary');

    this.options = {
      plugins: {
        legend: {
          labels: {
            color: textColor
          }
        }
      },
      scales: {
        r: {
          grid: {
            color: textColorSecondary
          },
          pointLabels: {
            color: textColorSecondary
          }
        }
      }
    };

    this.resulLoading = true;

    var juices: any = [];

    var extractedImages : any = [];

    var colors = ['--blue-400', '--pink-400', '--bluegray-400', '--green-400', '--red-400']

    this.http.get(this.url + "/results").subscribe(

      async (res: any) => {
        for (const face of res) {

          juices.push(
            {
              label: face.face_id.toUpperCase(),
              borderColor: documentStyle.getPropertyValue(colors[juices.length] ?? colors[0]),
              pointBackgroundColor: documentStyle.getPropertyValue(colors[juices.length] ?? colors[0]),
              pointBorderColor: documentStyle.getPropertyValue(colors[juices.length] ?? colors[0]),
              pointHoverBackgroundColor: textColor,
              pointHoverBorderColor: documentStyle.getPropertyValue(colors[juices.length] ?? colors[0]),
              data: [
                parseFloat(face.emotions['Anger'].slice(0, -1)),
                parseFloat(face.emotions['Contempt'].slice(0, -1)),
                parseFloat(face.emotions['Disgust'].slice(0, -1)),
                parseFloat(face.emotions['Fear'].slice(0, -1)),
                parseFloat(face.emotions['Happy'].slice(0, -1)),
                parseFloat(face.emotions['Neutral'].slice(0, -1)),
                parseFloat(face.emotions['Sad'].slice(0, -1)),
                parseFloat(face.emotions['Surprised'].slice(0, -1))
              ]
            }
          )

          extractedImages.push(face.image_path)

        }

          this.data = {
            labels: ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised'],
            datasets: [...juices]
          };

          this.results = res

        this.resulLoading = false;
      },
      (error) => {
        console.error('Error !!!', error);
        this.resulLoading = false;
      }
    );
  }

}
