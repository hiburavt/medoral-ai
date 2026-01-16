import { Download, AlertTriangle, CheckCircle, Activity, FileText } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ResultsProps {
  data: any;
  onDownloadPdf: () => void;
  bioData: any;
}

export default function Results({ data, onDownloadPdf, bioData }: ResultsProps) {
  const isHealthy = data.class === 'Healthy';
  const confidence = parseFloat(data.raw_score) > 0.5 ? data.raw_score : (1 - data.raw_score);
  const percentage = (confidence * 100).toFixed(2);

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      
      {/* Top Status Card */}
      <Card className={`border-l-8 ${isHealthy ? 'border-l-green-500 bg-green-50/50' : 'border-l-destructive bg-destructive/5'}`}>
        <CardContent className="p-6 flex items-center justify-between">
            <div className="flex items-center gap-4">
            <div className={`p-3 rounded-full ${isHealthy ? 'bg-green-100 text-green-600' : 'bg-red-100 text-destructive'}`}>
                {isHealthy ? <CheckCircle size={32} /> : <AlertTriangle size={32} />}
            </div>
            <div>
                <h2 className={`text-2xl font-bold ${isHealthy ? 'text-green-700' : 'text-destructive'}`}>
                    {isHealthy ? 'Low Risk Detected' : 'Suspicious Lesion Detected'}
                </h2>
                <p className="text-muted-foreground">AI Confidence: <span className="font-mono font-bold text-foreground text-lg">{percentage}%</span></p>
            </div>
            </div>
            <div className="text-right hidden sm:block">
            <p className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Classification</p>
            <p className="text-xl font-bold text-foreground">{data.class}</p>
            </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="analysis" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="analysis">AI Analysis</TabsTrigger>
          <TabsTrigger value="summary">Patient Summary</TabsTrigger>
        </TabsList>
        
        <TabsContent value="analysis" className="mt-6">
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                        <Activity size={18} className="text-primary"/> AI Attention Heatmap
                    </CardTitle>
                    <CardDescription>
                        Visual explanation of the AI's decision making process
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="aspect-square bg-muted rounded-lg overflow-hidden relative group">
                            {data.heatmap ? (
                            <img src={`data:image/png;base64,${data.heatmap}`} alt="Heatmap" className="w-full h-full object-contain" />
                            ) : (
                            <div className="flex items-center justify-center h-full text-muted-foreground">No Heatmap Available</div>
                            )}
                        </div>
                        <div className="flex flex-col justify-center space-y-4">
                            <div className="p-4 bg-muted/50 rounded-lg border border-border">
                                <h4 className="font-semibold mb-2">Interpretation Guide</h4>
                                <ul className="text-sm text-muted-foreground space-y-2 list-disc list-inside">
                                    <li><span className="text-red-500 font-bold">Red Areas:</span> High attention (Suspicious)</li>
                                    <li><span className="text-yellow-500 font-bold">Yellow Areas:</span> Moderate attention</li>
                                    <li><span className="text-blue-500 font-bold">Blue Areas:</span> Low attention (Healthy)</li>
                                </ul>
                            </div>
                            <p className="text-sm text-muted-foreground">
                                The heatmap highlights the regions of the oral scan that most influenced the DenseNet201 model's prediction. 
                                Always verify these findings with clinical examination.
                            </p>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </TabsContent>

        <TabsContent value="summary" className="mt-6">
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <FileText size={18} className="text-primary"/> Assessment Data
                    </CardTitle>
                    <CardDescription>Patient bio-data provided during assessment</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-2 gap-x-12 gap-y-6">
                        <div className="space-y-1">
                            <p className="text-sm text-muted-foreground">Age</p>
                            <p className="font-medium text-lg">{bioData.age}</p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-sm text-muted-foreground">Gender</p>
                            <p className="font-medium text-lg">{bioData.gender}</p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-sm text-muted-foreground">Tobacco Use</p>
                            <p className={`font-medium text-lg ${bioData.tobacco !== 'No' ? 'text-orange-600' : ''}`}>{bioData.tobacco}</p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-sm text-muted-foreground">Alcohol Consumption</p>
                            <p className={`font-medium text-lg ${bioData.alcohol !== 'No' ? 'text-orange-600' : ''}`}>{bioData.alcohol}</p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-sm text-muted-foreground">Symptoms Duration</p>
                            <p className="font-medium text-lg">{bioData.symptoms_duration}</p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-sm text-muted-foreground">Pain Level</p>
                            <div className="flex items-center gap-2">
                                <div className="w-full bg-secondary h-2 rounded-full overflow-hidden max-w-[100px]">
                                    <div className="bg-primary h-full" style={{ width: `${parseInt(bioData.pain_level) * 10}%` }}></div>
                                </div>
                                <span className="font-medium text-lg">{bioData.pain_level}/10</span>
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </TabsContent>
      </Tabs>

      <div className="flex justify-center pt-4">
        <Button 
            onClick={onDownloadPdf} 
            className="w-full md:w-auto md:min-w-[300px] h-12 text-lg gap-2 shadow-lg"
            size="lg"
        >
            <Download size={20} />
            Download Full Medical Report
        </Button>
      </div>
      <p className="text-[10px] text-center text-muted-foreground">
        *This report is generated by AI for educational purposes and is not a substitute for professional medical advice.
      </p>
    </div>
  );
}
