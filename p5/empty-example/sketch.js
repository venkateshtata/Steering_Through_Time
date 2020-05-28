
var txt = 0;
let angle=0;
var steering;


function setup() {
	frameRate(60); 
  // put setup code here
  
  steering = loadImage("./steer3.jpg");
  txt = loadStrings("./driving_dataset/data.txt")
  createCanvas(1920, 1080)
  angleMode(DEGREES);
  noStroke();

  
  i=0
  }


function draw() {


	

	if(txt!=0){

		loadImage("./driving_dataset/"+i+".jpg", img => {
			image(img,1100, 100);
			console.log(txt[i])

			push();
			ang = txt[i].split(" ")
			angle = ang[1]
			translate(1350,800)
			imageMode(CENTER);
			rotate(angle);
			image(steering, 0, 0);
			pop();

			
		})

	}


	

	i+=1




	
}