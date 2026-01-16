import React, { useCallback, useState } from 'react';
import { UploadCloud, Image as ImageIcon } from 'lucide-react';
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
}

export default function ImageUpload({ onImageSelect }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);
    onImageSelect(file);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  return (
    <Card className="max-w-md mx-auto shadow-sm hover:shadow-md transition-shadow">
      <CardContent className="p-0">
        <div 
            className={cn(
                "relative border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-all duration-300 min-h-[300px] flex flex-col items-center justify-center",
                isDragging ? "border-primary bg-primary/5 scale-[1.02]" : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50"
            )}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById('fileInput')?.click()}
        >
            <input 
            type="file" 
            id="fileInput" 
            className="hidden" 
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} 
            />
            
            {preview ? (
            <div className="relative group w-full h-full">
                <img 
                src={preview} 
                alt="Preview" 
                className="mx-auto max-h-64 rounded-lg shadow-md object-contain"
                />
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded-lg">
                <p className="text-white font-medium flex items-center gap-2">
                    <UploadCloud size={20} /> Change Image
                </p>
                </div>
            </div>
            ) : (
            <div className="space-y-4 py-8">
                <div className="bg-primary/10 w-20 h-20 rounded-full flex items-center justify-center mx-auto text-primary mb-4">
                <ImageIcon size={40} />
                </div>
                <h3 className="text-xl font-semibold text-foreground">Upload Image</h3>
                <p className="text-muted-foreground text-sm">
                Drag & drop or click to browse
                </p>
                <p className="text-xs text-muted-foreground/60 mt-2">
                Supports JPG, PNG, WEBP
                </p>
            </div>
            )}
        </div>
      </CardContent>
    </Card>
  );
}
