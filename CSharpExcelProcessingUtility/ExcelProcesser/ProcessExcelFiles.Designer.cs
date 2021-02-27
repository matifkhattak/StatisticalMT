namespace ExcelProcesser
{
    partial class ProcessExcelFiles
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.txtSingleFileReadLocation = new System.Windows.Forms.TextBox();
            this.btnSingleFile = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.lblWriteLocation = new System.Windows.Forms.Label();
            this.txtSingleFileWriteLocation = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.txtInstanceNo = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.btnExtractProbabilities = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.txtFrequencyDistributionFilesPath = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            // 
            // txtSingleFileReadLocation
            // 
            this.txtSingleFileReadLocation.Location = new System.Drawing.Point(149, 62);
            this.txtSingleFileReadLocation.Margin = new System.Windows.Forms.Padding(4);
            this.txtSingleFileReadLocation.Name = "txtSingleFileReadLocation";
            this.txtSingleFileReadLocation.Size = new System.Drawing.Size(1290, 22);
            this.txtSingleFileReadLocation.TabIndex = 0;
            // 
            // btnSingleFile
            // 
            this.btnSingleFile.Location = new System.Drawing.Point(314, 422);
            this.btnSingleFile.Margin = new System.Windows.Forms.Padding(4);
            this.btnSingleFile.Name = "btnSingleFile";
            this.btnSingleFile.Size = new System.Drawing.Size(289, 28);
            this.btnSingleFile.TabIndex = 1;
            this.btnSingleFile.Text = "Generate Frequency Distribution (RQ#1)";
            this.btnSingleFile.UseVisualStyleBackColor = true;
            this.btnSingleFile.Click += new System.EventHandler(this.btnSingleFile_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(11, 65);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(123, 17);
            this.label1.TabIndex = 2;
            this.label1.Text = "Read Location :";
            // 
            // lblWriteLocation
            // 
            this.lblWriteLocation.AutoSize = true;
            this.lblWriteLocation.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblWriteLocation.Location = new System.Drawing.Point(11, 215);
            this.lblWriteLocation.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.lblWriteLocation.Name = "lblWriteLocation";
            this.lblWriteLocation.Size = new System.Drawing.Size(123, 17);
            this.lblWriteLocation.TabIndex = 4;
            this.lblWriteLocation.Text = "Write Location :";
            // 
            // txtSingleFileWriteLocation
            // 
            this.txtSingleFileWriteLocation.Location = new System.Drawing.Point(149, 212);
            this.txtSingleFileWriteLocation.Margin = new System.Windows.Forms.Padding(4);
            this.txtSingleFileWriteLocation.Name = "txtSingleFileWriteLocation";
            this.txtSingleFileWriteLocation.Size = new System.Drawing.Size(1290, 22);
            this.txtSingleFileWriteLocation.TabIndex = 3;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(11, 138);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(73, 17);
            this.label2.TabIndex = 8;
            this.label2.Text = "Row No :";
            // 
            // txtInstanceNo
            // 
            this.txtInstanceNo.Location = new System.Drawing.Point(149, 134);
            this.txtInstanceNo.Margin = new System.Windows.Forms.Padding(4);
            this.txtInstanceNo.Name = "txtInstanceNo";
            this.txtInstanceNo.Size = new System.Drawing.Size(100, 22);
            this.txtInstanceNo.TabIndex = 7;
            this.txtInstanceNo.TextChanged += new System.EventHandler(this.textBox1_TextChanged);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.ForeColor = System.Drawing.Color.Crimson;
            this.label3.Location = new System.Drawing.Point(271, 138);
            this.label3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(693, 17);
            this.label3.TabIndex = 9;
            this.label3.Text = "(Row no, it represents the row number of recored to be processed, inclusive of to" +
    "p 2 rows acting as headers)";
            // 
            // btnExtractProbabilities
            // 
            this.btnExtractProbabilities.Location = new System.Drawing.Point(660, 422);
            this.btnExtractProbabilities.Margin = new System.Windows.Forms.Padding(4);
            this.btnExtractProbabilities.Name = "btnExtractProbabilities";
            this.btnExtractProbabilities.Size = new System.Drawing.Size(385, 28);
            this.btnExtractProbabilities.TabIndex = 19;
            this.btnExtractProbabilities.Text = "Extract Probabilities of Max Time Predicted Class (RQ#2)";
            this.btnExtractProbabilities.UseVisualStyleBackColor = true;
            this.btnExtractProbabilities.Click += new System.EventHandler(this.btnExtractProbabilities_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.Location = new System.Drawing.Point(16, 273);
            this.label4.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(902, 17);
            this.label4.TabIndex = 21;
            this.label4.Text = "Frequency Distribution Files Path For Source Execution(Used for RQ#2 only to get " +
    "the class which is predicted max times) :";
            // 
            // txtFrequencyDistributionFilesPath
            // 
            this.txtFrequencyDistributionFilesPath.Location = new System.Drawing.Point(19, 294);
            this.txtFrequencyDistributionFilesPath.Margin = new System.Windows.Forms.Padding(4);
            this.txtFrequencyDistributionFilesPath.Name = "txtFrequencyDistributionFilesPath";
            this.txtFrequencyDistributionFilesPath.Size = new System.Drawing.Size(1420, 22);
            this.txtFrequencyDistributionFilesPath.TabIndex = 20;
            // 
            // ProcessExcelFiles
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1471, 613);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.txtFrequencyDistributionFilesPath);
            this.Controls.Add(this.btnExtractProbabilities);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.txtInstanceNo);
            this.Controls.Add(this.lblWriteLocation);
            this.Controls.Add(this.txtSingleFileWriteLocation);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.btnSingleFile);
            this.Controls.Add(this.txtSingleFileReadLocation);
            this.Margin = new System.Windows.Forms.Padding(4);
            this.Name = "ProcessExcelFiles";
            this.Text = "Excel Processor";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox txtSingleFileReadLocation;
        private System.Windows.Forms.Button btnSingleFile;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label lblWriteLocation;
        private System.Windows.Forms.TextBox txtSingleFileWriteLocation;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox txtInstanceNo;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Button btnExtractProbabilities;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox txtFrequencyDistributionFilesPath;
    }
}

