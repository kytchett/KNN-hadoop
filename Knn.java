
import java.io.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Vector;
import java.lang.Math;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Knn{
	
	public static final String TRAINPATH = "knn_train.txt";
	public static final int K = 1;  // K nearest 
	public static final int MAXTYPE = 4;  // total number of types, values in [0,MAXTYPE)

	// training data
	private static List<Point> trainingPoint = new ArrayList<>();
	
	public static void readTrainingPoints(String path, String fileName){
		try {			
			FileReader fr = new FileReader(path + fileName);			
			BufferedReader bf = new BufferedReader(fr);			
			String str;
			// read file in lines		
			while ((str = bf.readLine()) != null) {	
				// new point in line			
				trainingPoint.add(new Knn.Point(str, 0));			
			}			
			bf.close();			
			fr.close();		
		} catch (IOException e) {			
			e.printStackTrace();		
		}
	}

        public static class KnnMapper extends Mapper<LongWritable, Text, Text, Text> {

                @Override
                protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			
			// new test data point
                	Point testPoint = new Knn.Point(value.toString(), 1);
			// compute distance of between testPoint and each training point
                    	for (Point eachTrainPoint : trainingPoint) {
                        	double dis = testPoint.getDistance(eachTrainPoint);
                        	context.write(new Text(testPoint.getPointText()), new Text( eachTrainPoint.getType() + "-" + dis));
                	}
        	}
	}

        public static class KnnReducer extends Reducer<Text, Text, Text, IntWritable> {
                @Override
                protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			// record each distanceTuple for point key
			List<DistanceTuple> list = new ArrayList<>();
			int value_cnt = 0;
			for (Text value : values) {
				// divide value into tuple(type, distance)
				String temp[] = value.toString().split("-");
				int valueType = Integer.parseInt(temp[0]);
				double valueDis = Double.parseDouble(temp[1]);
				list.add(new DistanceTuple(valueType, valueDis));
				value_cnt += 1;
			}
			// sort the tuple list
			Collections.sort(list);
			
			// record the times of each type happened in the k-nearest neighbour
			int[] count = new int[MAXTYPE];
			for (int i = 0; i < MAXTYPE; i++) {
				count[i] = 0;
			}
			for (int i = 0; i < K; i++) {
				count[list.get(i).getType()]++;
			}

			// find the type with the maximum of counts.
			int type = 0;
			int max_count = count[0];
			for(int i = 1; i < MAXTYPE; i ++){
				if(count[i] > max_count){
					max_count = count[i];
					type = i;				
				}
			}
			context.write(key, new IntWritable(type));

                }
        }


        public static void main(String[] args) throws Exception {
		// 读取训练数据
		Knn.readTrainingPoints("./", TRAINPATH);
                // 创建配置对象
                Configuration conf = new Configuration();
                // 创建Job对象
                Job job = Job.getInstance(conf, "Knn");
                // 设置运行Job的类
                job.setJarByClass(Knn.class);
                // 设置Mapper类
                job.setMapperClass(KnnMapper.class);
                // 设置Reducer类
                job.setReducerClass(KnnReducer.class);
                // 设置Map输出的Key value
                job.setMapOutputKeyClass(Text.class);
                job.setMapOutputValueClass(Text.class);
                // 设置Reduce输出的Key value
                job.setOutputKeyClass(Text.class);
                job.setOutputValueClass(IntWritable.class);
                // 设置输入输出的路径
                FileInputFormat.setInputPaths(job, new Path(args[0]));
                FileOutputFormat.setOutputPath(job, new Path(args[1]));
                // 提交job
                boolean b = job.waitForCompletion(true);

                if(!b) {
                        System.out.println("Knn task fail!");
                }

        }
	
	static class DistanceTuple  implements Comparable<DistanceTuple> {
	    int type;   // type
	    double distance;  // distance to the type
	    public DistanceTuple(int type, double distance){
		this.type = type;
		this.distance = distance;
	    }

	    public double getDistance(){
	    	return distance;
	    }

	    public int getType(){
	    	return type;
	    }
	    
	    // defination reducer to compare two tuple
	    @Override
	    public int compareTo(DistanceTuple dt) {
		return ((this.distance - dt.getDistance()) > 0 ? 1 : -1);
	    }
	    
	    @Override
	    public String toString() {
		return "" + type + ":" + distance;
	    }
	}

	// Point definition
	static class Point{
		private int type;  // type of this point
		private int dim;   // the dimension of the point
		private String pointText; // print in the result data 
		private String initStr; // for hadoop easy to transport between mapper and reducer
		private Vector<Double> value = new Vector<>();  // location detail

		public int getType(){
			return type;
		}

		public Vector<Double> getValue(){
			return value;	
		}
		
		Point(){
			type = -1;
		}		

		// initial point variable by a string
		// "dataType" points out whether this point is a test data or train data
		public Point(String str, int dataType){ 
			if(dataType == 0){
			// train data, the type is at the end of the str
				String temp[] = str.split(" ");
				int len = temp.length;
				for(int i = 0; i < len - 1; i ++){
					this.value.add(Double.parseDouble(temp[i])); // get the point location
				}
				this.type = Double.valueOf(temp[len - 1]).intValue(); // get the type
				this.dim = len - 1; // the dimension
				this.pointText = "(";
				for(int i = 0; i < len - 2; i ++){
					this.pointText = this.pointText + temp[i] + ",";
				}
				this.pointText = this.pointText + temp[len - 2] + ")"; // for reducer to output
				this.initStr = str; // transport between mapper and reducer
			}
			else if(dataType == 1){
			// test data, without type at the end
				String temp[] = str.split(" ");
				int len = temp.length;
				for(int i = 0; i < len; i ++){
					this.value.add(Double.parseDouble(temp[i])); // get the point location
				}
				this.type = -1; // no type
				this.dim = len; // the dimension
				this.pointText = "(";
				for(int i = 0; i < len - 1; i ++){
					this.pointText = this.pointText + temp[i] + ",";
				}
				this.pointText = this.pointText + temp[len - 1] + ")";// for reducer to output

				this.initStr = str; // transport between mapper and reducer
			}
			
		}

		
		public double getDistance(Point y){ // compute Euler Distance between this point and point y
			double dis = 0;
			Vector<Double> yValue = y.getValue();
			for(int i = 0; i < this.dim; i ++){
				dis += (this.value.elementAt(i) - yValue.elementAt(i)) * (this.value.elementAt(i) - yValue.elementAt(i));
			}
			dis = Math.sqrt(dis);
			return dis;
		}

		@Override
		public String toString() {
			return this.initStr;
		}
		
		public String getPointText(){
			return this.pointText;
		}

	}

}


