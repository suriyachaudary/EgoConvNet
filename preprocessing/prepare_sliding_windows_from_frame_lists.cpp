#include<stdio.h>
#include<string.h>
#include <cstdlib>
#define SLIDING_SIZE 30

char classes[11][20]={"close","pour","open","spread","scoop","take","fold", "shake", "put","stir","x"};
int get_label(char *a)
{
    for(int i=0;i<11;i++)
    {
        if(!strcmp(a,classes[i]))
            return i;
    }

    return 11;
}


int main(int argc, char *argv[])
{
	printf("\nargv[1] : file containing video names");
	printf("\nargv[2] : path to annotation files");
	printf("\nargv[3] : path to directory with frame list\n");
	printf("\nargv[4] : path to directory to place output files");
	printf("\nargv[5] : path to directory png_stab\n");
	printf("\nargv[6] : output annotation file \n");

	char *path_to_video_list = argv[1];
	char *path_to_annotation_files = argv[2];
	char *path_to_directory_with_frame_list = argv[3];

	FILE *fp_video =  fopen(path_to_video_list, "r");
	FILE *fp_dense_annotations = fopen(argv[6],"w");

	while(!feof(fp_video))
	{
		char video_name[300];
		fscanf(fp_video, "%s", video_name);

		
		printf("%s\n", video_name);

		char annotations_file[300];
		sprintf(annotations_file,"%s/%s.txt",path_to_annotation_files, video_name);
		FILE *fp_annotations = fopen(annotations_file,"r");
		printf("%s\n", annotations_file);
		char frame_list_file[300];
		sprintf(frame_list_file,"%s/%s.txt",path_to_directory_with_frame_list, video_name);
		
		FILE *fp_frame_list = fopen(frame_list_file,"r");

		
		int current_frame=1;

		while(!feof(fp_annotations))
		{
			char action[300], object[300];
			int start_frame = -1, end_frame = -1;
			fscanf(fp_annotations, "%s%s%d%d", action, object, &start_frame, &end_frame);

			if(!strcmp(action,"x"))
				break;

			while(current_frame < start_frame) //prepare windows - frames with no action label assigned to BG 'x' class
			{
				/*if(current_frame > start_frame - SLIDING_SIZE/2)
				{
					current_frame++;
					continue;
				}*/
					
				char frame[300];
				fscanf(fp_frame_list,"%s",frame);
				
				

				fprintf(fp_dense_annotations, "%s/%s_%s_%010d 10\n", argv[4],video_name, "x", current_frame);
				fprintf(stdout, "%s/%s_%s_%010d 10\n", argv[4],video_name, "x", current_frame);
				char output_file[300];
				sprintf(output_file,"%s/%s_%s_%010d", argv[4], video_name, "x", current_frame);

				FILE *fp_output = fopen(output_file,"w");
				for(int i=current_frame-SLIDING_SIZE/2; i<current_frame+SLIDING_SIZE/2;i+=1)
				{
					int j = abs(i);
					if(j>0)
					{
						fprintf(fp_output, "%s/%s/%010d.png\n", argv[5], video_name, j);
					}
				}

				

				fclose(fp_output);

				current_frame++;
			}

			while(current_frame >= start_frame  && current_frame<=end_frame) //prepare windows - frames with action annotations
			{
				/*if(current_frame < start_frame + SLIDING_SIZE/2)
				{
					current_frame++;
					continue;
				}
				
				if(current_frame > end_frame - SLIDING_SIZE/2)
				{
					current_frame++;
					continue;
				}*/
				
				char frame[300];
				fscanf(fp_frame_list,"%s",frame);

				
				fprintf(fp_dense_annotations, "%s/%s_%s_%010d %d\n", argv[4],video_name, action, current_frame,get_label(action));
				fprintf(stdout, "%s/%s_%s_%010d %d\n", argv[4],video_name, action, current_frame,get_label(action));
				char output_file[300];
				sprintf(output_file,"%s/%s_%s_%010d", argv[4], video_name, action, current_frame);
				FILE *fp_output = fopen(output_file,"w");
				for(int i=current_frame-SLIDING_SIZE/2; i<current_frame+SLIDING_SIZE/2;i+=1)
				{
					int j= abs(i);
					if(j>0)
					{
						fprintf(fp_output, "%s/%s/%010d.png\n", argv[5], video_name, j);
					}
				}

				

				fclose(fp_output);

				current_frame++;
			}
			
			/*for(int i=0;i<SLIDING_SIZE/2;i++)
				current_frame++;*/
			
		}

		fclose(fp_annotations);
		
		char frame[300];
		
		/*for(int i=0;i<SLIDING_SIZE/2 && !feof(fp_frame_list);i++)
				fscanf(fp_frame_list,"%s",frame);*/
				
		while(!feof(fp_frame_list)) //prepare windows - all remaining frames with no label at end of video - assign to BG 'x'
		{
			char frame[300];

			fscanf(fp_frame_list,"%s",frame);

			
				
			fprintf(fp_dense_annotations, "%s/%s_%s_%010d 10\n", argv[4],video_name, "x", current_frame);
			fprintf(stdout, "%s/%s_%s_%010d 10\n", argv[4],video_name, "x", current_frame);
			char output_file[300];
			sprintf(output_file,"%s/%s_%s_%010d",argv[4], video_name, "x", current_frame);
			FILE *fp_output = fopen(output_file,"w");
			for(int i=current_frame-SLIDING_SIZE/2; i<current_frame+SLIDING_SIZE/2;i+=1)
				{
					int j = abs(i);
					if(j>0)
					{
						fprintf(fp_output, "%s/%s/%010d.png\n", argv[5], video_name, j);
					}
				}

				
			fclose(fp_output);

			current_frame++;
		}

		
		fclose(fp_frame_list);
	}
	fclose(fp_dense_annotations);

	fclose(fp_video);

	return 0;
}
